package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.GpJSONOptionMap;
import it.necst.gpjson.engine.disk.SavedIndexBuilder;
import it.necst.gpjson.utils.UnsafeHelper;
import org.graalvm.polyglot.Value;

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.TimeUnit;
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
    private final int numLevels;
    private int numLines;

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
        this.numLevels = numLevels;
        this.build(combined);
    }

    public IndexBuilder(Value cu, Map<String, Value> kernels, DataBuilder dataBuilder, SavedIndexBuilder index) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.numLevels = index.getNumLevels();
        this.numLines = index.getNumLines();
        this.load(index);
        this.isIntermediateFreed = true;
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
        else {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Index already freed!");
        }
    }

    public Value getStringIndexMemory() {
        if (!isFreed)
            return stringIndexMemory;
        else {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Index already freed!");
        }
    }

    public Value getLeveledBitmapsIndexMemory() {
        if (!isFreed)
            return leveledBitmapsIndexMemory;
        else {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Index already freed!");
        }
    }

    public long getNumLevels() {
        if (!isFreed)
            return numLevels;
        else {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Index already freed!");
        }
    }

    public int getNumLines() {
        if (!isFreed)
            return numLines;
        else {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Index already freed!");
        }
    }

    private void load(SavedIndexBuilder index) {
        long start = System.nanoTime();
        newlineIndexMemory = byteArrayToDeviceArray(index.getNewlineIndexMemory());
        stringIndexMemory = byteArrayToDeviceArray(index.getStringIndexMemory());
        leveledBitmapsIndexMemory = byteArrayToDeviceArray(index.getLeveledBitmapsIndexMemory());
        LOGGER.log(Level.FINER, "load index done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    public SavedIndexBuilder save() {
        long start = System.nanoTime();
        byte[] newlineIndex = deviceArrayToByteArray(newlineIndexMemory);
        byte[] stringIndex = deviceArrayToByteArray(stringIndexMemory);
        byte[] leveledBitmapsIndex = deviceArrayToByteArray(leveledBitmapsIndexMemory);
        LOGGER.log(Level.FINER, "save index done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return new SavedIndexBuilder(newlineIndex, stringIndex, leveledBitmapsIndex, numLevels, numLines);
    }

    private Value byteArrayToDeviceArray(byte[] byteArray) {
        Value deviceArray = cu.invokeMember("DeviceArray", "long", byteArray.length / 8);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(byteArray.length);
        byteBuffer.put(byteArray);
        byteBuffer.position(0);
        deviceArray.invokeMember("copyFrom", UnsafeHelper.createLongArray(byteBuffer.asLongBuffer()).getAddress());
        return deviceArray;
    }

    private byte[] deviceArrayToByteArray(Value deviceArray) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) deviceArray.getArraySize() * 8);
        deviceArray.invokeMember("copyTo", UnsafeHelper.createLongArray(byteBuffer.asLongBuffer()).getAddress());
        byte[] byteArray = new byte[byteBuffer.remaining()];
        byteBuffer.get(byteArray);
        return byteArray;
    }

    private void build(boolean combined) {
        long start;
        start = System.nanoTime();
        this.createNewlineStringIndex(combined);
        LOGGER.log(Level.FINER, "createNewlineStringIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINEST, "newlineIndex has a size of " + (float) newlineIndexMemory.getArraySize()*4/1048576 + "MB");
        LOGGER.log(Level.FINEST, "stringIndex has a size of " + (float) stringIndexMemory.getArraySize()*8/1048576 + "MB");
        start = System.nanoTime();
        this.createLeveledBitmapsIndex();
        LOGGER.log(Level.FINER, "createLeveledBitmapsIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        LOGGER.log(Level.FINEST, "createLeveledBitmapsIndex has a size of " + (float) leveledBitmapsIndexMemory.getArraySize()*8/1048576 + "MB");
        
    }

    private void createNewlineStringIndex(boolean combined) {
        long start;
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize());
        stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        if (combined) {
            start = System.nanoTime();
            kernels.get("combined-escape-carry-newline-count-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, newlineCountIndexMemory);
            LOGGER.log(Level.FINEST, "combined-escape-carry-newline-count-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("newline-count-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), newlineCountIndexMemory);
            kernels.get("escape-carry-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory);
        }
        start = System.nanoTime();
        newlineIndexOffset = cu.invokeMember("DeviceArray", "int", gridSize * blockSize + 1);
        intSumBase = cu.invokeMember("DeviceArray", "int", reductionGridSize*reductionBlockSize);
        kernels.get("int-sum-pre-scan").execute(reductionGridSize, reductionBlockSize).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize());
        kernels.get("int-sum-post-scan").execute(1, 1).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, 1, intSumBase);
        kernels.get("int-sum-rebase").execute(reductionGridSize, reductionBlockSize).execute(newlineCountIndexMemory, newlineCountIndexMemory.getArraySize(), intSumBase, 1, newlineIndexOffset);
        newlineIndexOffset.setArrayElement(0, 1);
        numLines = newlineIndexOffset.getArrayElement(newlineIndexOffset.getArraySize()-1).asInt();
        LOGGER.log(Level.FINEST, "partial-sums() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", numLines);
        escapeIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize());
        if (combined) {
            start = System.nanoTime();
            kernels.get("combined-escape-newline-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, newlineIndexOffset, escapeIndexMemory, newlineIndexMemory);
            LOGGER.log(Level.FINEST, "combined-escape-newline-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } else {
            kernels.get("escape-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringCarryIndexMemory, escapeIndexMemory);
            kernels.get("newline-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), newlineIndexOffset, newlineIndexMemory);
        }
        start = System.nanoTime();
        kernels.get("quote-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory);
        LOGGER.log(Level.FINEST, "quote-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        xorBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("xor-pre-scan").execute(reductionGridSize, reductionBlockSize).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize());
        kernels.get("xor-post-scan").execute(1, 1).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, xorBase);
        kernels.get("xor-rebase").execute(reductionGridSize, reductionBlockSize).execute(stringCarryIndexMemory, stringCarryIndexMemory.getArraySize(), xorBase);
        LOGGER.log(Level.FINEST, "xor-partial-sums() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        kernels.get("string-index").execute(gridSize, blockSize).execute(stringIndexMemory, dataBuilder.getLevelSize(), stringCarryIndexMemory);
        LOGGER.log(Level.FINEST, "string-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    private void createLeveledBitmapsIndex() {
        long start;
        carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        start = System.nanoTime();
        kernels.get("leveled-bitmaps-carry-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringIndexMemory, carryIndexMemory);
        LOGGER.log(Level.FINEST, "leveled-bitmaps-carry-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        start = System.nanoTime();
        carryIndexMemoryWithOffset = cu.invokeMember("DeviceArray", "char", gridSize * blockSize + 1);
        carryIndexMemoryWithOffset.setArrayElement(0, -1);
        charSumBase = cu.invokeMember("DeviceArray", "char", reductionGridSize*reductionBlockSize);
        kernels.get("char-sum-pre-scan").execute(reductionGridSize, reductionBlockSize).execute(carryIndexMemory, carryIndexMemory.getArraySize());
        kernels.get("char-sum-post-scan").execute(1, 1).execute(carryIndexMemory, carryIndexMemory.getArraySize(), reductionGridSize*reductionBlockSize, -1, charSumBase);
        kernels.get("char-sum-rebase").execute(reductionGridSize, reductionBlockSize).execute(carryIndexMemory, carryIndexMemory.getArraySize(), charSumBase, 1, carryIndexMemoryWithOffset);
        LOGGER.log(Level.FINEST, "partial-sums() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", dataBuilder.getLevelSize() * numLevels);
        start = System.nanoTime();
        kernels.get("leveled-bitmaps-index").execute(gridSize, blockSize).execute(dataBuilder.getFileMemory(), dataBuilder.getFileSize(), stringIndexMemory, carryIndexMemoryWithOffset, leveledBitmapsIndexMemory, dataBuilder.getLevelSize(), numLevels);
        LOGGER.log(Level.FINEST, "leveled-bitmaps-index() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
