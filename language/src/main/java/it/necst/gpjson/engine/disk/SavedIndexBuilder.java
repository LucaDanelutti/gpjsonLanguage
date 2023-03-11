package it.necst.gpjson.engine.disk;

import java.io.Serializable;

public class SavedIndexBuilder implements Serializable {
    private final byte[] newlineIndexMemory;
    private final byte[] stringIndexMemory;
    private final byte[] leveledBitmapsIndexMemory;
    private final long numLevels;
    private final int numLines;

    public SavedIndexBuilder(byte[] newlineIndexMemory, byte[] stringIndexMemory, byte[] leveledBitmapsIndexMemory, long numLevels, int numLines) {
        this.newlineIndexMemory = newlineIndexMemory;
        this.stringIndexMemory = stringIndexMemory;
        this.leveledBitmapsIndexMemory = leveledBitmapsIndexMemory;
        this.numLevels = numLevels;
        this.numLines = numLines;
    }

    public byte[] getNewlineIndexMemory() {
        return newlineIndexMemory;
    }

    public byte[] getStringIndexMemory() {
        return stringIndexMemory;
    }

    public byte[] getLeveledBitmapsIndexMemory() {
        return leveledBitmapsIndexMemory;
    }

    public long getNumLevels() {
        return numLevels;
    }

    public int getNumLines() {
        return numLines;
    }
}
