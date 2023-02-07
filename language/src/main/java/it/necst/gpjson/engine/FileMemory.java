package it.necst.gpjson.engine;

import org.graalvm.polyglot.Value;
import java.nio.MappedByteBuffer;

public class FileMemory {
    private final Value cu;
    private final String fileName;
    private Value fileMemory;
    private final MappedByteBuffer fileBuffer;
    private final long fileSize;
    private final long levelSize;

    public FileMemory(Value cu, String fileName, MappedByteBuffer fileBuffer, long fileSize) {
        this.cu = cu;
        this.fileName = fileName;
        this.fileBuffer = fileBuffer;
        this.fileSize = fileSize;
        this.load();
        this.levelSize = (fileMemory.getArraySize() + 64 - 1) / 64;
    }

    public void free() {
        fileMemory.invokeMember("free");
    }

    public Value getFileMemory() {
        return fileMemory;
    }

    public long getLevelSize() {
        return levelSize;
    }

    public long getFileSize() {
        return fileSize;
    }

    public MappedByteBuffer getFileBuffer() {
        return fileBuffer;
    }

    public String getFileName() {
        return fileName;
    }

    private void load() {
        fileMemory = cu.invokeMember("DeviceArray", "char", fileSize);
        UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(fileBuffer);
        fileMemory.invokeMember("copyFrom", byteArray.getAddress());
    }
}
