package it.necst.gpjson.engine;

import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.MyLogger;
import it.necst.gpjson.jsonpath.*;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

public abstract class ExecutionContext {
    protected final Value cu;
    protected final Map<String,Value> kernels;
    protected final int gridSize = 8; //8 or 512
    protected final int blockSize = 1024;
    private final String fileName;
    private MappedByteBuffer fileBuffer;
    protected Value fileMemory;
    private boolean isLoaded = false;
    protected Value newlineIndexMemory;
    protected Value stringIndexMemory;
    protected Value leveledBitmapsIndexMemory;
    protected long levelSize;
    protected long numLevels;
    private boolean isIndexed = false;

    public ExecutionContext(Value cu, Map<String,Value> kernels, String fileName) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileName = fileName;
    }

    public MappedByteBuffer getFileBuffer() {
        return fileBuffer;
    }

    public void loadFile() {
        long start;
        start = System.nanoTime();
        Path file = Paths.get(fileName);
        long fileSize;
        try {
            fileSize = Files.size(file);
        } catch (IOException e) {
            throw new GpJSONException("Failed to get file size");
        }
        fileMemory = cu.invokeMember("DeviceArray", "char", fileSize);
        levelSize = (fileMemory.getArraySize() + 64 - 1) / 64;
        try (FileChannel channel = FileChannel.open(file)) {
            if (channel.size() != fileSize) {
                throw new GpJSONException("Size of file has changed while reading");
            }
            fileBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            fileBuffer.load();
            UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(fileBuffer);
            fileMemory.invokeMember("copyFrom", byteArray.getAddress());
        } catch (IOException e) {
            throw new GpJSONException("Failed to open file");
        }
        MyLogger.log(Level.FINER, "ExecutionContext", "loadFile()", "loadFile() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        isLoaded = true;
    }

    public void buildIndexes(long numLevels) {
        if (!isLoaded)
            this.loadFile();
        this.numLevels = numLevels;
        long start;
        start = System.nanoTime();
        this.createNewlineStringIndex();
        MyLogger.log(Level.FINER, "ExecutionContext", "buildIndexes()", "createNewlineStringIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        this.createLeveledBitmapsIndex();
        MyLogger.log(Level.FINER, "ExecutionContext", "buildIndexes()", "createLeveledBitmapsIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        isIndexed = true;
    }

    public long[][] execute(String query) throws UnsupportedJSONPathException {
        long start = System.nanoTime();
        JSONPathResult compiledQuery = this.compileQuery(query);
        MyLogger.log(Level.FINER, "ExecutionContext", "execute()", "compileQuery() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        if (!isIndexed || numLevels < compiledQuery.getMaxDepth())
            this.buildIndexes(compiledQuery.getMaxDepth());
        start = System.nanoTime();
        long[][] resultIndexes = this.query(compiledQuery);
        MyLogger.log(Level.FINER, "ExecutionContext", "execute()", "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }

    public List<List<String>> executeAndGetStrings(String query) throws UnsupportedJSONPathException {
        long[][] resultIndexes = execute(query);
        long start = System.nanoTime();
        List<List<String>> resultsString = this.fetchResults(resultIndexes);
        MyLogger.log(Level.FINER, "ExecutionContext", "executeAndGetStrings()", "fetchResults() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultsString;
    }

    private JSONPathResult compileQuery(String query) {
        try {
            return new JSONPathParser(new JSONPathScanner(query)).compile();
        } catch (JSONPathException e) {
            throw new GpJSONException("Error parsing query: " + query);
        }
    }

    protected abstract void createNewlineStringIndex();

    private void createLeveledBitmapsIndex() {
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize * numLevels);
        kernels.get("initialize").execute(gridSize, blockSize).execute(leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), 0);
        Value carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        kernels.get("create_leveled_bitmaps_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemory);
        int level = -1;
        for (int i=0; i<carryIndexMemory.getArraySize(); i++) {
            int value = carryIndexMemory.getArrayElement(i).asInt();
            carryIndexMemory.setArrayElement(i, level);
            level += value;
        }
        kernels.get("create_leveled_bitmaps").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemory, leveledBitmapsIndexMemory, levelSize * numLevels, levelSize, numLevels);
    }

    private long[][] query(JSONPathResult compiledQuery) {
        long numberOfLines = newlineIndexMemory.getArraySize();
        long numberOfResults = compiledQuery.getNumResults();
        Value result = cu.invokeMember("DeviceArray", "long", numberOfLines * 2 * numberOfResults);
        Value queryMemory = cu.invokeMember("DeviceArray", "char", compiledQuery.getIr().size());
        kernels.get("initialize").execute(gridSize, blockSize).execute(result, result.getArraySize(), -1);
        byte[] queryByteArray = compiledQuery.getIr().toByteArray();
        for (int j = 0; j < queryByteArray.length; j++) {
            queryMemory.setArrayElement(j, queryByteArray[j]);
        }
        kernels.get("find_value").execute(512, 1024).execute(fileMemory, fileMemory.getArraySize(), newlineIndexMemory, newlineIndexMemory.getArraySize(), stringIndexMemory, leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), levelSize, queryMemory, compiledQuery.getNumResults(), result);
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(result.getArraySize());
        result.invokeMember("copyTo", longArray.getAddress());
        long[][] resultIndexes = new long[(int) numberOfLines][(int) numberOfResults * 2];
        for (int j = 0; j < numberOfLines; j++) {
            for (int k = 0; k < compiledQuery.getNumResults()*2; k+=2) {
                resultIndexes[j][k] = longArray.getValueAt(j*numberOfResults*2+ k);
                resultIndexes[j][k+1] = longArray.getValueAt(j*numberOfResults*2 + k + 1);
            }
        }
        return resultIndexes;
    }

    private List<List<String>> fetchResults(long[][] resultIndexes) {
        List<List<String>> resultStrings = new ArrayList<>();
        for (long[] resultIndex : resultIndexes) {
            List<String> toAdd = new ArrayList<>();
            for (int j = 0; j < resultIndexes[0].length; j += 2) {
                long valueStart = resultIndex[j];
                if (valueStart == -1)
                    continue;
                long valueEnd = resultIndex[j + 1];
                byte[] value = new byte[(int) (valueEnd - valueStart)];
                fileBuffer.position((int) valueStart);
                fileBuffer.get(value);
                toAdd.add(new String(value, StandardCharsets.UTF_8));
            }
            resultStrings.add(toAdd);
        }
        return resultStrings;
    }
}
