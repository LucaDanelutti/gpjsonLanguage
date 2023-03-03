package it.necst.gpjson.engine;

import it.necst.gpjson.jsonpath.JSONPathResult;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.polyglot.Value;
import it.necst.gpjson.GpJSONLogger;

import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class FileQuery {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int queryGridSize = 512;
    private final int queryBlockSize = 1024;

    private Value resultMemory;

    private final FileMemory fileMemory;
    private final FileIndex fileIndex;
    private final JSONPathResult compiledQuery;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    private Value queryMemory;

    public FileQuery(Value cu, Map<String, Value> kernels, FileMemory fileMemory, FileIndex fileIndex, JSONPathResult query) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileMemory = fileMemory;
        this.fileIndex = fileIndex;
        this.compiledQuery = query;
        this.query();
    }

    public void free() {
        queryMemory.invokeMember("free");

        resultMemory.invokeMember("free");
    }

    public int[][] copyBuildResultArray() {
        long localStart;
        long start = System.nanoTime();
        long numberOfResults = compiledQuery.getNumResults();
        localStart = System.nanoTime();
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(resultMemory.getArraySize());
        resultMemory.invokeMember("copyTo", longArray.getAddress());
        LOGGER.log(Level.FINEST, "copyTo() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        localStart = System.nanoTime();
        int[][] resultIndexes = new int[fileIndex.getNumLines()][(int) numberOfResults * 2];
        for (int j = 0; j < fileIndex.getNumLines(); j++) {
            for (int k = 0; k < numberOfResults*2; k+=2) {
                resultIndexes[j][k] = (int) longArray.getValueAt(j*numberOfResults*2+ k);
                resultIndexes[j][k+1] = (int) longArray.getValueAt(j*numberOfResults*2 + k + 1);
            }
        }
        LOGGER.log(Level.FINEST, "resultIndexes() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "copyBuildResultArray() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }

    private void query() {
        long localStart;
        long start = System.nanoTime();
        long numberOfResults = compiledQuery.getNumResults();
        queryMemory = cu.invokeMember("DeviceArray", "char", compiledQuery.getIr().size());
        localStart = System.nanoTime();
        byte[] queryByteArray = compiledQuery.getIr().toByteArray();
        StringBuilder stringBuilder = new StringBuilder();
        for (int j = 0; j < queryByteArray.length; j++) {
            queryMemory.setArrayElement(j, queryByteArray[j]);
            stringBuilder.append(String.format("%x | ", Byte.toUnsignedInt(queryByteArray[j])));
        }
        LOGGER.log(Level.FINER, "compiledQuery: " + stringBuilder);
        LOGGER.log(Level.FINEST, "copyCompiledQuery() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        resultMemory = cu.invokeMember("DeviceArray", "long", fileIndex.getNumLines() * 2 * numberOfResults);
        localStart = System.nanoTime();
        kernels.get("find_value").execute(queryGridSize, queryBlockSize).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), fileIndex.getNewlineIndexMemory(), fileIndex.getNumLines(), fileIndex.getStringIndexMemory(), fileIndex.getLeveledBitmapsIndexMemory(), fileMemory.getLevelSize()*fileIndex.getNumLevels(), fileMemory.getLevelSize(), queryMemory, compiledQuery.getNumResults(), resultMemory);
        LOGGER.log(Level.FINEST, "find_value() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
