package it.necst.gpjson.engine.core;

import it.necst.gpjson.GpJSONOptionMap;
import it.necst.gpjson.utils.UnsafeHelper;
import it.necst.gpjson.jsonpath.JSONPathQuery;
import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.polyglot.Value;
import it.necst.gpjson.GpJSONLogger;

import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class QueryExecutor {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int gridSize = GpJSONOptionMap.getQueryExecutorGridSize();
    private final int blockSize = GpJSONOptionMap.getQueryExecutorBlockSize();

    private Value resultMemory;

    private final DataBuilder dataBuilder;
    private final IndexBuilder indexBuilder;
    private final JSONPathQuery compiledQuery;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    private Value queryMemory;

    public QueryExecutor(Value cu, Map<String, Value> kernels, DataBuilder dataBuilder, IndexBuilder indexBuilder, JSONPathQuery query) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.indexBuilder = indexBuilder;
        this.compiledQuery = query;
        this.query();
    }

    public void free() {
        long localStart = System.nanoTime();
        queryMemory.invokeMember("free");
        resultMemory.invokeMember("free");
        LOGGER.log(Level.FINEST, "free(queryExecutor) done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");

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
        int[][] resultIndexes = new int[indexBuilder.getNumLines()][(int) numberOfResults * 2];
        for (int j = 0; j < indexBuilder.getNumLines(); j++) {
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
            if (queryByteArray[j] >= '!' && queryByteArray[j] < 'z')
                stringBuilder.append(" | " + (char) queryByteArray[j]);
            else
                stringBuilder.append(" | " + queryByteArray[j]);
        }
        LOGGER.log(Level.FINEST, "copyCompiledQuery() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINEST, "compiledQuery: " + stringBuilder.toString());
        resultMemory = cu.invokeMember("DeviceArray", "long", indexBuilder.getNumLines() * 2 * numberOfResults);
        localStart = System.nanoTime();
        kernels.get("query").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), indexBuilder.getNewlineIndexMemory(), indexBuilder.getNumLines(), indexBuilder.getStringIndexMemory(), indexBuilder.getLeveledBitmapsIndexMemory(), dataBuilder.getLevelSize()* indexBuilder.getNumLevels(), dataBuilder.getLevelSize(), queryMemory, compiledQuery.getNumResults(), resultMemory);
        LOGGER.log(Level.FINEST, "find_value() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
