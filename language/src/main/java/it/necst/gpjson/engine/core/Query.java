package it.necst.gpjson.engine.core;

import it.necst.gpjson.engine.UnsafeHelper;
import it.necst.gpjson.jsonpath.JSONPathQuery;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.polyglot.Value;
import it.necst.gpjson.GpJSONLogger;

import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class Query {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int queryGridSize = 512;
    private final int queryBlockSize = 1024;

    private Value resultMemory;

    private final Data data;
    private final Index index;
    private final JSONPathQuery compiledQuery;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    private Value queryMemory;

    public Query(Value cu, Map<String, Value> kernels, Data data, Index index, JSONPathQuery query) {
        this.cu = cu;
        this.kernels = kernels;
        this.data = data;
        this.index = index;
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
        int[][] resultIndexes = new int[index.getNumLines()][(int) numberOfResults * 2];
        for (int j = 0; j < index.getNumLines(); j++) {
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
        resultMemory = cu.invokeMember("DeviceArray", "long", index.getNumLines() * 2 * numberOfResults);
        localStart = System.nanoTime();
        kernels.get("find_value").execute(queryGridSize, queryBlockSize).execute(data.getFileMemory(), data.getFileSize(), index.getNewlineIndexMemory(), index.getNumLines(), index.getStringIndexMemory(), index.getLeveledBitmapsIndexMemory(), data.getLevelSize()* index.getNumLevels(), data.getLevelSize(), queryMemory, compiledQuery.getNumResults(), resultMemory);
        LOGGER.log(Level.FINEST, "find_value() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
