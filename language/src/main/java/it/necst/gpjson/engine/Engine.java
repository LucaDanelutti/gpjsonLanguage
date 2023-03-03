package it.necst.gpjson.engine;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.*;
import it.necst.gpjson.engine.core.Index;
import it.necst.gpjson.engine.core.Data;
import it.necst.gpjson.engine.core.Query;
import it.necst.gpjson.jsonpath.*;
import it.necst.gpjson.kernel.GpJSONKernel;
import it.necst.gpjson.result.Result;
import it.necst.gpjson.result.ResultGPJSONQuery;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

@ExportLibrary(InteropLibrary.class)
public class Engine implements TruffleObject {
    private static final String BUILDKERNELS = "buildKernels";
    private static final String QUERY = "query";
    private static final String LOAD = "load";
    private static final String INDEX = "index";
    private static final String QUERY2 = "query2";
    private static final String CLOSE = "close";

    private static final Set<String> MEMBERS = new HashSet<>(Arrays.asList(BUILDKERNELS, QUERY, LOAD, INDEX, QUERY2, CLOSE));

    private final Context polyglot;
    private final Value cu;
    Map<String,Value> kernels = new HashMap<>();

    private final int partitionSize = (int) (1 * Math.pow(2, 30));

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Engine() {
        polyglot = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("grcuda.ExecutionPolicy", "async")
                .option("grcuda.InputPrefetch", "true")
                .option("grcuda.RetrieveNewStreamPolicy", "always-new") // always-new, reuse
                .option("grcuda.RetrieveParentStreamPolicy", "multigpu-disjoint") // same-as-parent, disjoint, multigpu-early-disjoint, multigpu-disjoint
                .option("grcuda.DependencyPolicy", "with-const")
                .option("grcuda.DeviceSelectionPolicy", "min-transfer-size")
                .option("grcuda.ForceStreamAttach", "false")
                .option("grcuda.EnableComputationTimers", "false")
                .option("grcuda.MemAdvisePolicy", "none") // none, read-mostly, preferred
                .option("grcuda.NumberOfGPUs", "2")
                // DAG
                .option("grcuda.ExportDAG", "./dag")
                // logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "FINER")
                .option("log.grcuda.com.nvidia.grcuda.runtime.executioncontext.level", "FINEST")
                .build();
        LOGGER.log(Level.FINE, "grcuda context created");
        cu = polyglot.eval("grcuda", "CU");
        LOGGER.log(Level.FINE, "Engine created");
    }

    private void buildKernels() {
        if (kernels.isEmpty()) {
            long start;
            start = System.nanoTime();
            for (GpJSONKernel kernel : GpJSONKernel.values()) {
                try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream(kernel.getFilename())) {
                    if (inputStream != null) {
                        String code;
                        byte[] targetArray = new byte[inputStream.available()];
                        if (inputStream.read(targetArray) <= 0)
                            throw new GpJSONInternalException("error reading from " + kernel.getFilename());
                        code = new String(targetArray, StandardCharsets.UTF_8);
                        kernels.put(kernel.getName(), cu.invokeMember("buildkernel", code, kernel.getParameterSignature()));
                    } else {
                        throw new GpJSONInternalException("file not found " + kernel.getFilename());
                    }
                } catch (IOException e) {
                    throw new GpJSONInternalException("cannot read from " + kernel.getFilename());
                }
            }
            LOGGER.log(Level.FINER, "buildKernels() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        }
    }

    private Result query(String fileName, String[] queries, boolean combined, boolean batched) {
        if (kernels.isEmpty()) buildKernels();
        if (batched) {
            ResultGPJSONQuery query = queryBatch(fileName, queries[0], combined);
            Result result = new Result();
            result.addQuery(query);
            return result;
        } else {
            return queryBlock(fileName, queries, combined);
        }
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS.toArray(new String[0]);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return MEMBERS.contains(member);
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments) throws UnknownIdentifierException, UnsupportedTypeException {
        switch (member) {
            case BUILDKERNELS:
                if (arguments.length != 0) {
                    throw new GpJSONException(BUILDKERNELS + " function requires 0 arguments");
                }
                this.buildKernels();
                return this;
            case QUERY: {
                if ((arguments.length != 4)) {
                    throw new GpJSONException(QUERY + " function requires 4 arguments");
                }
                String file = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                String[] queries = InvokeUtils.expectStringArray(arguments[1], "argument 2 of " + QUERY + " must be an array of strings");
                boolean combined = InvokeUtils.expectBoolean(arguments[2], "argument 3 of " + QUERY + " must be a boolean");
                boolean batched = InvokeUtils.expectBoolean(arguments[3], "argument 4 of " + QUERY + " must be a boolean");
                return this.query(file, queries, combined, batched);
            case CLOSE:
                polyglot.close();
                return this;
            default:
                throw UnknownIdentifierException.create(member);
        }
    }

    private Data loadFileBlock(String fileName) {
        long start;
        start = System.nanoTime();
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
        Data file = new Data(cu, fileName, fileBuffer, fileSize);
        LOGGER.log(Level.FINER, "loadFile() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return file;
    }

    private Result queryBlock(String fileName, String[] queries, boolean combined) {
        Data data = loadFileBlock(fileName);
        JSONPathQuery[] compiledQueries = new JSONPathQuery[queries.length];
        int maxDepth = 0;
        for (int i=0; i< queries.length; i++) {
            try {
                compiledQueries[i] = new JSONPathParser(new JSONPathScanner(queries[i])).compile();
                maxDepth = Math.max(maxDepth, compiledQueries[i].getMaxDepth());
            } catch (UnsupportedJSONPathException e) {
                LOGGER.log(Level.FINE, "Unsupported JSONPath query '" + queries[i] + "'. Falling back to cpu execution.");
                compiledQueries[i] = null;
            } catch (JSONPathException e) {
                throw new GpJSONException("Error parsing query: " + queries[i]);
            }
        }
        Index index = new Index(cu, kernels, data, combined, maxDepth);
        Result result = new Result();
        for (int i = 0; i < compiledQueries.length; i++) {
            String query = queries[i];
            if (compiledQueries[i] != null) {
                Query fileQuery = new Query(cu, kernels, data, index, compiledQueries[i]);
                result.addQuery(fileQuery.copyBuildResultArray(), data.getFileBuffer());
                fileQuery.free();
                LOGGER.log(Level.FINE, query + " executed successfully");
            } else {
                result.addFallbackQuery(FileFallbackQuery.fallbackQuery(fileName, query));
                LOGGER.log(Level.FINE, query + " executed successfully (cpu fallback)");
            }
        }
        long start = System.nanoTime();
        data.free();
        index.free();
        LOGGER.log(Level.FINER, "free() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return result;
    }

    private ResultGPJSONQuery queryBatch(String fileName, String query, boolean combined) {
        long start;
        JSONPathQuery compiledQuery;
        try {
            compiledQuery = new JSONPathParser(new JSONPathScanner(query)).compile();
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
            MappedByteBuffer[] fileBuffer = new MappedByteBuffer[partitions.size()];
            Data[] data = new Data[partitions.size()];
            Index[] indices = new Index[partitions.size()];
            Query[] fileQuery = new Query[partitions.size()];
            LOGGER.log(Level.FINE, "Generated " + partitions.size() + " partitions (partition size = " + partitionSize + ")");
            LOGGER.log(Level.FINER, "partitions: " + partitions);

            ResultGPJSONQuery result = new ResultGPJSONQuery();
            int stride = 10;
            for (int j=0; j <= partitions.size() / stride; j++) {
                for (int i=j*stride; i < (j+1)*stride && i < partitions.size(); i++) {
                    start = System.nanoTime();
                    long startIndex = partitions.get(i);
                    long endIndex = (i == partitions.size()-1) ? fileSize : partitions.get(i+1) - 1; //skip the newline character
                    fileBuffer[i] = channel.map(FileChannel.MapMode.READ_ONLY, startIndex, endIndex-startIndex);
                    fileBuffer[i].load();
                    data[i] = new Data(cu, fileName, fileBuffer[i], endIndex-startIndex);
                    indices[i] = new Index(cu, kernels, data[i], combined, compiledQuery.getMaxDepth());
                    fileQuery[i] = new Query(cu, kernels, data[i], indices[i], compiledQuery);
                    long localStart = System.nanoTime();
                    data[i].free();
                    indices[i].free();
                    LOGGER.log(Level.FINER, "Memory and index of partition " + i + " freed in " + (System.nanoTime() - localStart) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
                    LOGGER.log(Level.FINER, "Partition " + i + " processed in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
                }

                for (int i=j*stride; i < (j+1)*stride && i < partitions.size(); i++) {
                    int[][] lines = fileQuery[i].copyBuildResultArray();
                    result.addPartition(lines, fileBuffer[i], indices[i].getNumLines());
                    start = System.nanoTime();
                    fileQuery[i].free();
                    LOGGER.log(Level.FINER, "Partition " + i + " freed in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
                    LOGGER.log(Level.FINE, "Partition " + i + " executed successfully");
                }
            }
            return result;
        } catch (IOException e) {
            throw new GpJSONException("Failed to open file");
        }
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
