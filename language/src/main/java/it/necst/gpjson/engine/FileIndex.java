package it.necst.gpjson.engine;

import com.oracle.truffle.api.TruffleLogger;
import org.graalvm.polyglot.Value;
import it.necst.gpjson.GpJSONLogger;

import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class FileIndex {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int gridSize = 1024*16;
    private final int blockSize = 1024;
    private final int reductionGridSize = 32;
    private final int reductionBlockSize = 32;

    private Value newlineIndexMemory;
    private Value stringIndexMemory;
    private Value leveledBitmapsIndexMemory;
    private final long numLevels;
    private int numLines;
    private final boolean combined;

    private final FileMemory fileMemory;

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

    public FileIndex(Value cu, Map<String, Value> kernels, FileMemory fileMemory, boolean combined, int numLevels) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileMemory = fileMemory;
        this.combined = combined;
        this.numLevels = numLevels;
        this.build();
    }

    public void free() {
        stringCarryIndexMemory.invokeMember("free");
        newlineCountIndexMemory.invokeMember("free");
        newlineIndexOffset.invokeMember("free");
        intSumBase.invokeMember("free");
        escapeIndexMemory.invokeMember("free");
        xorBase.invokeMember("free");

        carryIndexMemory.invokeMember("free");
        carryIndexMemoryWithOffset.invokeMember("free");
        charSumBase.invokeMember("free");

        newlineIndexMemory.invokeMember("free");
        stringIndexMemory.invokeMember("free");
        leveledBitmapsIndexMemory.invokeMember("free");
    }

    public Value getNewlineIndexMemory() {
        return newlineIndexMemory;
    }

    public Value getStringIndexMemory() {
        return stringIndexMemory;
    }

    public Value getLeveledBitmapsIndexMemory() {
        return leveledBitmapsIndexMemory;
    }

    public long getNumLevels() {
        return numLevels;
    }

    public int getNumLines() {
        return numLines;
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
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", fileMemory.getLevelSize());
        stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_carry_newline_count_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringCarryIndexMemory, newlineCountIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_carry_newline_count_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("count_newlines").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), newlineCountIndexMemory);
            kernels.get("create_escape_carry_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringCarryIndexMemory);
        }
        start = System.nanoTime();
        newlineIndexOffset = cu.invokeMember("DeviceArray", "int", gridSize * blockSize + 1);
        intSumBase = cu.invokeMember("DeviceArray", "int", reductionGridSize*reductionBlockSize);
        kernels.get("int_sum1").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize());
        kernels.get("int_sum2").execute(1, 1, fileMemory.getStream()).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, 1, intSumBase);
        kernels.get("int_sum3").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), intSumBase, 1, newlineIndexOffset);
        cu.invokeMember("cudaStreamSynchronize", fileMemory.getStream());
        newlineIndexOffset.setArrayElement(0, 1);
        numLines = newlineIndexOffset.getArrayElement(newlineIndexOffset.getArraySize()-1).asInt();
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", numLines);
        escapeIndexMemory = cu.invokeMember("DeviceArray", "long", fileMemory.getLevelSize());
        if (combined) {
            start = System.nanoTime();
            kernels.get("create_combined_escape_newline_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringCarryIndexMemory, newlineIndexOffset, escapeIndexMemory, fileMemory.getLevelSize(), newlineIndexMemory);
            LOGGER.log(Level.FINEST, "create_combined_escape_newline_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("create_escape_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringCarryIndexMemory, escapeIndexMemory, fileMemory.getLevelSize());
            kernels.get("create_newline_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), newlineIndexOffset, newlineIndexMemory);
        }
        start = System.nanoTime();
        kernels.get("create_quote_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory, fileMemory.getLevelSize());
        LOGGER.log(Level.FINEST, "create_quote_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        xorBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("xor1").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize());
        kernels.get("xor2").execute(1, 1, fileMemory.getStream()).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, xorBase);
        kernels.get("xor3").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), xorBase);
        LOGGER.log(Level.FINEST, "xor() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        kernels.get("create_string_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getLevelSize(), stringIndexMemory, stringCarryIndexMemory);
        LOGGER.log(Level.FINEST, "create_string_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void createLeveledBitmapsIndex() {
        long start;
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", fileMemory.getLevelSize() * numLevels);
        start = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize, fileMemory.getStream()).execute(leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), 0);
        LOGGER.log(Level.FINEST, "initialize() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps_carry_index").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringIndexMemory, carryIndexMemory);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps_carry_index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        carryIndexMemoryWithOffset = cu.invokeMember("DeviceArray", "char", gridSize * blockSize + 1);
        carryIndexMemoryWithOffset.setArrayElement(0, -1);
        charSumBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("char_sum1").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(carryIndexMemory, carryIndexMemory.getArraySize());
        kernels.get("char_sum2").execute(1, 1, fileMemory.getStream()).execute(carryIndexMemory, carryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, -1, charSumBase);
        kernels.get("char_sum3").execute(reductionGridSize, reductionBlockSize, fileMemory.getStream()).execute(carryIndexMemory, carryIndexMemory.getArraySize(), charSumBase, 1, carryIndexMemoryWithOffset);
        LOGGER.log(Level.FINEST, "sum() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        kernels.get("create_leveled_bitmaps").execute(gridSize, blockSize, fileMemory.getStream()).execute(fileMemory.getFileMemory(), fileMemory.getFileSize(), stringIndexMemory, carryIndexMemoryWithOffset, leveledBitmapsIndexMemory, fileMemory.getLevelSize() * numLevels, fileMemory.getLevelSize(), numLevels);
        LOGGER.log(Level.FINEST, "create_leveled_bitmaps() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
