package it.necst.gpjson.engine;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.jsonpath.JSONPathResult;
import org.graalvm.polyglot.Value;

import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class Executor {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int gridSize = 8; //8 or 512
    private final int blockSize = 1024;

    private Value newlineIndexMemory;
    private Value stringIndexMemory;
    private Value leveledBitmapsIndexMemory;
    private long numLevels;
    private boolean isIndexed = false;
    private boolean combined = false;

    private final Value fileMemory;
    private final long levelSize;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Executor(Value cu, Map<String, Value> kernels, Value fileMemory, boolean combined) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileMemory = fileMemory;
        this.levelSize = (fileMemory.getArraySize() + 64 - 1) / 64;
        this.combined = combined;
    }

    public void buildIndexes(long numLevels) {
        if (!isIndexed || numLevels > this.numLevels) {
            this.numLevels = numLevels;
            long start;
            start = System.nanoTime();
            this.createNewlineStringIndex();
            LOGGER.log(Level.FINER, "createNewlineStringIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            start = System.nanoTime();
            this.createLeveledBitmapsIndex();
            LOGGER.log(Level.FINER, "createLeveledBitmapsIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            isIndexed = true;
        }
    }

    private void createLeveledBitmapsIndex() {
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize * numLevels);
        long startInitialize = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize).execute(leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), 0);
        LOGGER.log(Level.FINEST, "initialize done in " + (System.nanoTime() - startInitialize) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
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

    private void createNewlineStringIndex() {
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        Value stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        Value newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        if (combined)
            kernels.get("create_combined_escape_carry_newline_count_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, newlineCountIndexMemory);
        else {
            kernels.get("count_newlines").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineCountIndexMemory);
            kernels.get("create_escape_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory);
        }
        int sum = 1;
        for (int i=0; i<newlineCountIndexMemory.getArraySize(); i++) {
            int val = newlineCountIndexMemory.getArrayElement(i).asInt();
            newlineCountIndexMemory.setArrayElement(i, sum);
            sum += val;
        }
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", sum);
        Value escapeIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        if (combined)
            kernels.get("create_combined_escape_newline_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, newlineCountIndexMemory, escapeIndexMemory, levelSize, newlineIndexMemory);
        else {
            kernels.get("create_escape_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, escapeIndexMemory, levelSize);
            kernels.get("create_newline_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineCountIndexMemory, newlineIndexMemory);
        }
        kernels.get("create_quote_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory, levelSize);
        byte prev = 0;
        for (int i=0; i<stringCarryIndexMemory.getArraySize(); i++) {
            byte value = (byte) (stringCarryIndexMemory.getArrayElement(i).asByte() ^ prev);
            stringCarryIndexMemory.setArrayElement(i, value);
            prev = value;
        }
        kernels.get("create_string_index").execute(gridSize, blockSize).execute(levelSize, stringIndexMemory, stringCarryIndexMemory);
    }

    public long[][] query(JSONPathResult compiledQuery) {
        if (!isIndexed)
            throw new GpJSONInternalException("You must index the file before querying");
        long start = System.nanoTime();
        long numberOfLines = newlineIndexMemory.getArraySize();
        long numberOfResults = compiledQuery.getNumResults();
        Value result = cu.invokeMember("DeviceArray", "long", numberOfLines * 2 * numberOfResults);
        Value queryMemory = cu.invokeMember("DeviceArray", "char", compiledQuery.getIr().size());
        long startInitialize = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize).execute(result, result.getArraySize(), -1);
        LOGGER.log(Level.FINEST, "initialize done in " + (System.nanoTime() - startInitialize) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        byte[] queryByteArray = compiledQuery.getIr().toByteArray();
        for (int j = 0; j < queryByteArray.length; j++) {
            queryMemory.setArrayElement(j, queryByteArray[j]);
        }
        kernels.get("find_value").execute(512, 1024).execute(fileMemory, fileMemory.getArraySize(), newlineIndexMemory, newlineIndexMemory.getArraySize(), stringIndexMemory, leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), stringIndexMemory.getArraySize(), queryMemory, compiledQuery.getNumResults(), result);
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(result.getArraySize());
        result.invokeMember("copyTo", longArray.getAddress());
        long[][] resultIndexes = new long[(int) numberOfLines][(int) numberOfResults * 2];
        for (int j = 0; j < numberOfLines; j++) {
            for (int k = 0; k < compiledQuery.getNumResults()*2; k+=2) {
                resultIndexes[j][k] = longArray.getValueAt(j*numberOfResults*2+ k);
                resultIndexes[j][k+1] = longArray.getValueAt(j*numberOfResults*2 + k + 1);
            }
        }
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }
}
