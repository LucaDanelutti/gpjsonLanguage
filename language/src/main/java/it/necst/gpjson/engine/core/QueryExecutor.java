package it.necst.gpjson.engine.core;

import it.necst.gpjson.utils.UnsafeHelper;
import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONLogger;
import org.graalvm.polyglot.Value;

import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class QueryExecutor {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int queryGridSize = 1024;
    private final int queryBlockSize = 512;

    private Value resultMemory;
    private Value resultOffsetMemory;

    private final DataBuilder dataBuilder;
    private final IndexBuilder indexBuilder;
    private final QueryCompiler queryCompiler;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    private Value queryMemory;
    private Value queryOffsetMemory;

    public QueryExecutor(Value cu, Map<String, Value> kernels, DataBuilder dataBuilder, IndexBuilder indexBuilder, QueryCompiler queryCompiler) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.indexBuilder = indexBuilder;
        this.queryCompiler = queryCompiler;
        this.query();
    }

    public void free() {
        long localStart = System.nanoTime();
        queryMemory.invokeMember("free");
        queryOffsetMemory.invokeMember("free");
        resultMemory.invokeMember("free");
        resultOffsetMemory.invokeMember("free");
        LOGGER.log(Level.FINEST, "free(queryExecutor) done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    public int[][][] copyBuildResultArray() {
        long localStart;
        long start = System.nanoTime();
        localStart = System.nanoTime();
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(resultMemory.getArraySize());
        resultMemory.invokeMember("copyTo", longArray.getAddress());
        LOGGER.log(Level.FINEST, "copyTo() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        localStart = System.nanoTime();
        int[][][] resultIndexes = new int[queryCompiler.getNumQueries()][][];
        long resultOffset = 0;
        for (int i = 0; i < queryCompiler.getNumQueries(); i++) {
            int numResults = queryCompiler.getCompiledQuery(i).getNumResults();
            resultIndexes[i] = new int[indexBuilder.getNumLines()][numResults * 2];
            for (int j = 0; j < indexBuilder.getNumLines(); j++) {
                for (int k = 0; k < numResults*2; k+=2) {
                    resultIndexes[i][j][k] = (int) longArray.getValueAt(resultOffset*2*indexBuilder.getNumLines() + (long) j*numResults*2 + k);
                    resultIndexes[i][j][k+1] = (int) longArray.getValueAt(resultOffset*2*indexBuilder.getNumLines() + (long) j*numResults*2 + k + 1);
                }
            }
            resultOffset += numResults;
        }
        LOGGER.log(Level.FINEST, "resultIndexes() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "copyBuildResultArray() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }

    private void query() {
        long localStart;
        long start = System.nanoTime();
        localStart = System.nanoTime();
        copyCompiledQueries();
        LOGGER.log(Level.FINEST, "copyCompiledQuery() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        resultMemory = cu.invokeMember("DeviceArray", "long", indexBuilder.getNumLines() * 2 * queryCompiler.getTotalNumResults());
        localStart = System.nanoTime();
        kernels.get("executeQuery").execute(queryGridSize, queryBlockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), indexBuilder.getNewlineIndexMemory(), indexBuilder.getNumLines(), indexBuilder.getStringIndexMemory(), indexBuilder.getLeveledBitmapsIndexMemory(), dataBuilder.getLevelSize(), queryMemory, queryOffsetMemory, queryCompiler.getNumQueries(), resultOffsetMemory, resultMemory);
        LOGGER.log(Level.FINEST, "find_value() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void copyCompiledQueries() {
        queryOffsetMemory = cu.invokeMember("DeviceArray", "int", queryCompiler.getNumQueries());
        resultOffsetMemory = cu.invokeMember("DeviceArray", "int", queryCompiler.getNumQueries()+1);
        queryMemory = cu.invokeMember("DeviceArray", "char", queryCompiler.getTotalSize());
        int queryOffset = 0;
        int resultOffset = 0;
        for (int i = 0; i < queryCompiler.getNumQueries(); i++) {
            queryOffsetMemory.setArrayElement(i,queryOffset);
            resultOffsetMemory.setArrayElement(i, resultOffset);
            resultOffset += queryCompiler.getCompiledQuery(i).getNumResults();
            byte[] queryByteArray = queryCompiler.getCompiledQuery(i).getIr().toByteArray();
            for (byte b: queryByteArray) {
                queryMemory.setArrayElement(queryOffset, b);
                queryOffset++;
            }
        }
        resultOffsetMemory.setArrayElement(queryCompiler.getNumQueries(), resultOffset);
    }
}
