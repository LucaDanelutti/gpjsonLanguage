package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.utils.UnsafeHelper;
import org.graalvm.polyglot.Value;

import java.nio.MappedByteBuffer;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class DataBuilder {
    private final Value cu;
    private final String fileName;
    private Value fileMemory;
    private final MappedByteBuffer fileBuffer;
    private final int fileSize;
    private final int levelSize;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public DataBuilder(Value cu, String fileName, MappedByteBuffer fileBuffer, int fileSize) {
        this.cu = cu;
        this.fileName = fileName;
        this.fileBuffer = fileBuffer;
        this.fileSize = fileSize;
        this.levelSize = (fileSize + 64 - 1) / 64;
        long localStart = System.nanoTime();
        fileMemory = cu.invokeMember("DeviceArray", "char", fileSize);
        LOGGER.log(Level.FINEST, "createDeviceArray() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }

    public void free() {
        fileMemory.invokeMember("free");
    }

    public Value getFileMemory() {
        return fileMemory;
    }

    public int getLevelSize() {
        return levelSize;
    }

    public int getFileSize() {
        return fileSize;
    }

    public MappedByteBuffer getFileBuffer() {
        return fileBuffer;
    }

    public String getFileName() {
        return fileName;
    }

    public void load() {
        long localStart = System.nanoTime();
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(fileBuffer);
        LOGGER.log(Level.FINEST, "createByteArray() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        localStart = System.nanoTime();
        fileMemory.invokeMember("copyFrom", byteArray.getAddress());
        LOGGER.log(Level.FINEST, "copyFrom() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
    }
}
