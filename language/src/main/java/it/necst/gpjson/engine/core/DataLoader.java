package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class DataLoader {
    private final Value cu;
    private final Map<String,Value> kernels;

    private final int partitionSize;
    private final DataBuilder[] dataBuilder;
    private int numPartitions;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public DataLoader(Value cu, Map<String, Value> kernels, String fileName, int partitionSize) {
        this.cu = cu;
        this.kernels = kernels;
        this.partitionSize = partitionSize;
        if (partitionSize == 0)
            this.dataBuilder = new DataBuilder[]{loadFile(fileName)};
        else
            this.dataBuilder = loadFileBatch(fileName);
    }

    public DataBuilder[] getDataBuilder() {
        return dataBuilder;
    }

    public int getNumPartitions() {
        return numPartitions;
    }

    private DataBuilder[] loadFileBatch(String fileName) {
        Path file = Paths.get(fileName);
        long fileSize;
        DataBuilder[] dataBuilder;
        try {
            fileSize = Files.size(file);
        } catch (IOException e) {
            throw new GpJSONException("Failed to get file size");
        }
        try (FileChannel channel = FileChannel.open(file)) {
            if (channel.size() != fileSize) {
                throw new GpJSONException("Size of file has changed while reading");
            }
            List<Long> partitions = new ArrayList<>();
            long lastPartition = 0;
            partitions.add(lastPartition);
            while (fileSize - lastPartition > partitionSize) {
                partitions.add(nextPartition(channel, lastPartition));
                lastPartition = partitions.get(partitions.size() - 1);
            }
            numPartitions = partitions.size();
            LOGGER.log(Level.FINE, "Generated " + numPartitions + " partitions (partition size = " + partitionSize + ")");
            MappedByteBuffer[] fileBuffer = new MappedByteBuffer[partitions.size()];
            dataBuilder = new DataBuilder[partitions.size()];
            for (int i=0; i < numPartitions; i++) {
                cu.invokeMember("cudaSetDevice", i % 2);
                long startIndex = partitions.get(i);
                long endIndex = (i == partitions.size()-1) ? fileSize : partitions.get(i+1) - 1; //skip the newline character
                fileBuffer[i] = channel.map(FileChannel.MapMode.READ_ONLY, startIndex, endIndex-startIndex);
                fileBuffer[i].load();
                dataBuilder[i] = new DataBuilder(cu, fileName, fileBuffer[i], endIndex-startIndex);
            }
        } catch (IOException e) {
            throw new GpJSONException("Failed to open file");
        }
        cu.invokeMember("cudaSetDevice", 0);
        return dataBuilder;
    }

    private DataBuilder loadFile(String fileName) {
        long start = System.nanoTime();
        long fileSize;
        Path filePath = Paths.get(fileName);
        try {
            fileSize = Files.size(filePath);
            if (fileSize > Integer.MAX_VALUE)
                throw new GpJSONException("Block mode cannot process files > 2GB");
        } catch (IOException e) {
            throw new GpJSONException("Failed to get file size");
        }
        MappedByteBuffer fileBuffer;
        try (FileChannel channel = FileChannel.open(filePath)) {
            if (channel.size() != fileSize) {
                throw new GpJSONException("Size of file has changed while reading");
            }
            long localStart = System.nanoTime();
            fileBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            fileBuffer.load();
            LOGGER.log(Level.FINEST, "loadChannel() done in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        } catch (IOException e) {
            throw new GpJSONException("Failed to open file");
        }
        DataBuilder dataBuilder = new DataBuilder(cu, fileName, fileBuffer, fileSize);
        numPartitions = 1;
        LOGGER.log(Level.FINER, "loadFile() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return dataBuilder;
    }

    private long nextPartition(FileChannel channel, long prevPartition) {
        int offset = 0;
        while (offset < partitionSize) {
            ByteBuffer dest = ByteBuffer.allocate(1);
            try {
                channel.read(dest, prevPartition + partitionSize + offset);
            } catch (IOException e) {
                throw new GpJSONException("Failed to read from file");
            }
            if (dest.get(0) == '\n')
                return prevPartition + partitionSize + offset + 1; //we want the first character, not the newline
            offset--;
        }
        throw new GpJSONException("Cannot partition file");
    }
}
