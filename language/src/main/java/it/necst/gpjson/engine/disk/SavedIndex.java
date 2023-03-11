package it.necst.gpjson.engine.disk;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.GpJSONLogger;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class SavedIndex implements Serializable {
    private final SavedIndexBuilder[] savedIndexBuilders;
    private final int partitionSize;
    private final String inputFileHash;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public SavedIndex(SavedIndexBuilder[] savedIndexBuilders, int partitionSize, String inputFileHash) {
        this.savedIndexBuilders = savedIndexBuilders;
        this.partitionSize = partitionSize;
        this.inputFileHash = inputFileHash;
    }

    public int getPartitionSize() {
        return partitionSize;
    }

    public String getInputFileHash() {
        return inputFileHash;
    }

    public int getNumPartitions() {
        return savedIndexBuilders.length;
    }

    public SavedIndexBuilder[] getSavedIndexBuilders() {
        return savedIndexBuilders;
    }

    public void save(String fileName) {
        try {
            long start = System.nanoTime();
            FileOutputStream f = new FileOutputStream(fileName);
            ObjectOutputStream o = new ObjectOutputStream(f);
            o.writeObject(this);
            o.close();
            f.close();
            LOGGER.log(Level.FINER, "write index file done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } catch (FileNotFoundException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("File not found");
        } catch (IOException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("Failed to open file");
        }
    }

    public static SavedIndex restore(String fileName) {
        try {
            long start = System.nanoTime();
            FileInputStream fi = new FileInputStream(fileName);
            ObjectInputStream oi = new ObjectInputStream(fi);
            SavedIndex savedIndex = (SavedIndex) oi.readObject();
            LOGGER.log(Level.FINER, "read index file done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            return savedIndex;
        } catch (FileNotFoundException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("File not found");
        } catch (IOException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("Failed to open file");
        } catch (ClassNotFoundException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("Class not found");
        }
    }
}
