package it.necst.gpjson.engine;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.jsonpath.*;
import it.necst.gpjson.result.ResultGPJSONQuery;
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

public class BatchedExecutionContext {
    private final Value cu;
    private final Map<String,Value> kernels;
    private final int partitionSize;

    //File
    private final String fileName;
    private MappedByteBuffer[] fileBuffer;
    private Value[] fileMemory;
    private long levelSize;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public BatchedExecutionContext(Value cu, Map<String,Value> kernels, String fileName, int partitionSize) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileName = fileName;
        this.partitionSize = partitionSize;
    }

    private JSONPathResult compileQuery(String query) throws JSONPathException {
        long start = System.nanoTime();
        JSONPathResult result;
        result = new JSONPathParser(new JSONPathScanner(query)).compile();
        LOGGER.log(Level.FINER, "compileQuery() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return result;
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

    public ResultGPJSONQuery query(String query, boolean combined) {
        JSONPathResult compiledQuery;
        try {
            compiledQuery = this.compileQuery(query);
        } catch (UnsupportedJSONPathException e) {
            LOGGER.log(Level.FINE, "Unsupported JSONPath query '" + query + "'. Falling back to cpu execution");
            //TODO
            throw new GpJSONException("Error parsing query: " + query);
        } catch (JSONPathException e) {
            throw new GpJSONException("Error parsing query: " + query);
        }

        Path file = Paths.get(fileName);
        long fileSize;
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
                lastPartition = partitions.get(partitions.size()-1);
            }
            fileBuffer = new MappedByteBuffer[partitions.size()];
            fileMemory = new Value[partitions.size()];
            LOGGER.log(Level.FINE, "Generated " + partitions.size() + " partitions (partition size = " + partitionSize + ")");
            LOGGER.log(Level.FINER, "partitions: " + partitions);

            ResultGPJSONQuery result = new ResultGPJSONQuery();
            for (int i=0; i < partitions.size(); i++) {
                long start = System.nanoTime();
                long startIndex = partitions.get(i);
                long endIndex = (i == partitions.size()-1) ? fileSize : partitions.get(i+1) - 1; //skip the newline character
                fileBuffer[i] = channel.map(FileChannel.MapMode.READ_ONLY, startIndex, endIndex-startIndex);
                fileBuffer[i].load();
                UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(fileBuffer[i]);
                fileMemory[i] = cu.invokeMember("DeviceArray", "char", endIndex-startIndex);
                fileMemory[i].invokeMember("copyFrom", byteArray.getAddress());
                Executor ex = new Executor(cu, kernels, fileMemory[i], combined);
                ex.buildIndexes(compiledQuery.getMaxDepth());
                int[][] lines = ex.query(compiledQuery);
                result.addPartition(lines, fileBuffer[i], ex.getCountNewlines());
                LOGGER.log(Level.FINER, "partition " + i + " processed in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            }
            return result;
        } catch (IOException e) {
            throw new GpJSONException("Failed to open file");
        }
    }
}

