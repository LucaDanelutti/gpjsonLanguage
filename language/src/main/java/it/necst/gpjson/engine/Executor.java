package it.necst.gpjson.engine;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
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
    private final int gridSize = 512;
    private final int blockSize = 1024;
    private final int queryGridSize = 512;
    private final int queryBlockSize = 1024;

    private Value newlineIndexMemory;
    private Value stringIndexMemory;
    private Value leveledBitmapsIndexMemory;
    private long numLevels;
    private boolean isIndexed = false;
    private final boolean combined;

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
        long start;
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize * numLevels);
        start = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize).execute(leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), 0);
        LOGGER.log(Level.FINEST, "initialize done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        Value carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemory);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps_carry_index done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        Value carryIndexMemoryWithOffset = cu.invokeMember("DeviceArray", "char", gridSize * blockSize + 1);
        Value sumBase = cu.invokeMember("DeviceArray", "char", 32*32);
        kernels.get("char_sum1").execute(32,32).execute(carryIndexMemory, carryIndexMemory.getArraySize());
        kernels.get("char_sum2").execute(1,1).execute(carryIndexMemory, carryIndexMemory.getArraySize(), 32*32, -1, sumBase);
        kernels.get("char_sum3").execute(32,32).execute(carryIndexMemory, carryIndexMemory.getArraySize(), sumBase, 1, carryIndexMemoryWithOffset);
        carryIndexMemoryWithOffset.setArrayElement(0, -1);
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemoryWithOffset, leveledBitmapsIndexMemory, levelSize * numLevels, levelSize, numLevels);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void createNewlineStringIndex() {
        long start;
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        Value stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        Value newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_carry_newline_count_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, newlineCountIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_carry_newline_count_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("count_newlines").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineCountIndexMemory);
            kernels.get("create_escape_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory);
        }
        start = System.nanoTime();
        Value newlineIndexOffset = cu.invokeMember("DeviceArray", "int", gridSize * blockSize + 1);
        Value sumBase = cu.invokeMember("DeviceArray", "int", 32*32);
        kernels.get("int_sum1").execute(32,32).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize());
        kernels.get("int_sum2").execute(1,1).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), 32*32, 1, sumBase);
        kernels.get("int_sum3").execute(32,32).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), sumBase, 1, newlineIndexOffset);
        newlineIndexOffset.setArrayElement(0, 1);
        int sum = newlineIndexOffset.getArrayElement(newlineIndexOffset.getArraySize()-1).asInt();
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", sum);
        Value escapeIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_newline_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, newlineIndexOffset, escapeIndexMemory, levelSize, newlineIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_newline_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("create_escape_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, escapeIndexMemory, levelSize);
            kernels.get("create_newline_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineIndexOffset, newlineIndexMemory);
        }
        kernels.get("create_quote_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory, levelSize);
        start = System.nanoTime();
        Value xorBase = cu.invokeMember("DeviceArray", "char", 32*32);
        kernels.get("xor1").execute(32,32).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize());
        kernels.get("xor2").execute(1,1).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), 32*32, xorBase);
        kernels.get("xor3").execute(32,32).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), xorBase);
        LOGGER.log(Level.FINEST, "xor() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        kernels.get("create_string_index").execute(gridSize, blockSize).execute(levelSize, stringIndexMemory, stringCarryIndexMemory);
    }

    public int[][] query(JSONPathResult compiledQuery) {
        if (!isIndexed)
            throw new GpJSONException("You must index the file before querying");
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
        kernels.get("find_value").execute(queryGridSize, queryBlockSize).execute(fileMemory, fileMemory.getArraySize(), newlineIndexMemory, newlineIndexMemory.getArraySize(), stringIndexMemory, leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), stringIndexMemory.getArraySize(), queryMemory, compiledQuery.getNumResults(), result);
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(result.getArraySize());
        result.invokeMember("copyTo", longArray.getAddress());
        int[][] resultIndexes = new int[(int) numberOfLines][(int) numberOfResults * 2];
        for (int j = 0; j < numberOfLines; j++) {
            for (int k = 0; k < compiledQuery.getNumResults()*2; k+=2) {
                resultIndexes[j][k] = (int) longArray.getValueAt(j*numberOfResults*2+ k);
                resultIndexes[j][k+1] = (int) longArray.getValueAt(j*numberOfResults*2 + k + 1);
            }
        }
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }

    public int getCountNewlines() {
        return (int) newlineIndexMemory.getArraySize();
    }
}
