package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.GpJSONOptionMap;
import org.graalvm.polyglot.Value;
import it.necst.gpjson.GpJSONLogger;

import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class IndexBuilder {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int gridSize = GpJSONOptionMap.getIndexBuilderGridSize();
    private final int blockSize = GpJSONOptionMap.getIndexBuilderBlockSize();
    private final int reductionGridSize = 32;
    private final int reductionBlockSize = 32;

    private Value newlineIndexMemory;
    private Value stringIndexMemory;
    private Value leveledBitmapsIndexMemory;
    private final long numLevels;
    private int numLines;
    private final boolean combined;

    private final DataBuilder dataBuilder;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    private Value stringCarryIndexMemory;
    private Value newlineCountIndexMemory;
    private Value newlineIndexOffset;
    private Value intSumBase;
    private Value escapeIndexMemory;
    private Value xorBase;

    private Value carryIndexMemory;
    private Value carryIndexMemoryWithOffset;
    private Value charSumBase;

    private boolean isIntermediateFreed = false;
    private boolean isFreed = false;

    public IndexBuilder(Value cu, Map<String, Value> kernels, DataBuilder dataBuilder, boolean combined, int numLevels) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.combined = combined;
        this.numLevels = numLevels;
        this.build();
    }

    public void intermediateFree() {
        if (!isIntermediateFreed) {
            stringCarryIndexMemory.invokeMember("free");
            newlineCountIndexMemory.invokeMember("free");
            newlineIndexOffset.invokeMember("free");
            intSumBase.invokeMember("free");
            escapeIndexMemory.invokeMember("free");
            xorBase.invokeMember("free");

            carryIndexMemory.invokeMember("free");
            carryIndexMemoryWithOffset.invokeMember("free");
            charSumBase.invokeMember("free");

            isIntermediateFreed = true;
        }
    }

    public void free() {
        if (!isFreed) {
            newlineIndexMemory.invokeMember("free");
            stringIndexMemory.invokeMember("free");
            leveledBitmapsIndexMemory.invokeMember("free");

            isFreed = true;
        }
    }

    public Value getNewlineIndexMemory() {
        if (!isFreed)
            return newlineIndexMemory;
        else
            throw new GpJSONInternalException("Index already freed!");
    }

    public Value getStringIndexMemory() {
        if (!isFreed)
            return stringIndexMemory;
        else
            throw new GpJSONInternalException("Index already freed!");
    }

    public Value getLeveledBitmapsIndexMemory() {
        if (!isFreed)
            return leveledBitmapsIndexMemory;
        else
            throw new GpJSONInternalException("Index already freed!");
    }

    public long getNumLevels() {
        if (!isFreed)
            return numLevels;
        else
            throw new GpJSONInternalException("Index already freed!");
    }

    public int getNumLines() {
        if (!isFreed)
            return numLines;
        else
            throw new GpJSONInternalException("Index already freed!");
    }

    private void build() {
        long start;
        start = System.nanoTime();
        this.createNewlineStringIndex();
        LOGGER.log(Level.FINER, "createNewlineStringIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        this.createLeveledBitmapsIndex();
        LOGGER.log(Level.FINER, "createLeveledBitmapsIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void createNewlineStringIndex() {
        long start;
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize());
        stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_carry_newline_count_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, newlineCountIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_carry_newline_count_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("count_newlines").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), newlineCountIndexMemory);
            kernels.get("create_escape_carry_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory);
        }
        start = System.nanoTime();
        newlineIndexOffset = cu.invokeMember("DeviceArray", "int", gridSize * blockSize + 1);
        intSumBase = cu.invokeMember("DeviceArray", "int", reductionGridSize*reductionBlockSize);
        kernels.get("int_sum1").execute(reductionGridSize, reductionBlockSize).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize());
        kernels.get("int_sum2").execute(1, 1).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, 1, intSumBase);
        kernels.get("int_sum3").execute(reductionGridSize, reductionBlockSize).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), intSumBase, 1, newlineIndexOffset);
        newlineIndexOffset.setArrayElement(0, 1);
        numLines = newlineIndexOffset.getArrayElement(newlineIndexOffset.getArraySize()-1).asInt();
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", numLines);
        escapeIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize());
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_newline_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, newlineIndexOffset, escapeIndexMemory, dataBuilder.getLevelSize(), newlineIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_newline_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("create_escape_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, escapeIndexMemory, dataBuilder.getLevelSize());
            kernels.get("create_newline_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), newlineIndexOffset, newlineIndexMemory);
        }
        start = System.nanoTime();
        kernels.get("create_quote_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory, dataBuilder.getLevelSize());
        LOGGER.log(Level.FINEST, "create_quote_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        xorBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("xor1").execute(reductionGridSize, reductionBlockSize).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize());
        kernels.get("xor2").execute(1, 1).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, xorBase);
        kernels.get("xor3").execute(reductionGridSize, reductionBlockSize).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), xorBase);
        LOGGER.log(Level.FINEST, "xor() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        kernels.get("create_string_index").execute(gridSize, blockSize).execute(dataBuilder.getLevelSize(), stringIndexMemory, stringCarryIndexMemory);
        LOGGER.log(Level.FINEST, "create_string_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void createLeveledBitmapsIndex() {
        long start;
        carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps_carry_index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringIndexMemory, carryIndexMemory);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps_carry_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        carryIndexMemoryWithOffset = cu.invokeMember("DeviceArray", "char", gridSize * blockSize + 1);
        carryIndexMemoryWithOffset.setArrayElement(0, -1);
        charSumBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("char_sum1").execute(reductionGridSize, reductionBlockSize).execute(carryIndexMemory, carryIndexMemory.getArraySize());
        kernels.get("char_sum2").execute(1, 1).execute(carryIndexMemory, carryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, -1, charSumBase);
        kernels.get("char_sum3").execute(reductionGridSize, reductionBlockSize).execute(carryIndexMemory, carryIndexMemory.getArraySize(), charSumBase, 1, carryIndexMemoryWithOffset);
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize() * numLevels);
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringIndexMemory, carryIndexMemoryWithOffset, leveledBitmapsIndexMemory, dataBuilder.getLevelSize() * numLevels, dataBuilder.getLevelSize(), numLevels);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
