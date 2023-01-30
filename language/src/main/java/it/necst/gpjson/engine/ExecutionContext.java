package it.necst.gpjson.engine;

import com.jayway.jsonpath.*;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.InvokeUtils;
import it.necst.gpjson.jsonpath.*;
import it.necst.gpjson.result.Result;
import it.necst.gpjson.result.ResultFallbackQuery;
import it.necst.gpjson.result.ResultGPJSONQuery;
import it.necst.gpjson.result.ResultQuery;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

@ExportLibrary(InteropLibrary.class)
public abstract class ExecutionContext implements TruffleObject {
    private static final String LOADFILE = "loadFile";
    private static final String BUILDINDEXES = "buildIndexes";
    private static final String QUERY = "query";

    protected final Value cu;
    protected final Map<String,Value> kernels;
    protected final int gridSize = 8; //8 or 512
    protected final int blockSize = 1024;

    //File
    private final String fileName;
    private MappedByteBuffer fileBuffer;
    protected Value fileMemory;
    protected long levelSize;
    private boolean isLoaded = false;

    //Indexes
    protected Value newlineIndexMemory;
    protected Value stringIndexMemory;
    protected Value leveledBitmapsIndexMemory;
    protected long numLevels;
    private boolean isIndexed = false;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public ExecutionContext(Value cu, Map<String,Value> kernels, String fileName) {
        this.cu = cu;
        this.kernels = kernels;
        this.fileName = fileName;
    }

    private void loadFile() {
        if (!isLoaded) {
            long start;
            start = System.nanoTime();
            Path file = Paths.get(fileName);
            long fileSize;
            try {
                fileSize = Files.size(file);
            } catch (IOException e) {
                throw new GpJSONException("Failed to get file size");
            }
            fileMemory = cu.invokeMember("DeviceArray", "char", fileSize);
            levelSize = (fileMemory.getArraySize() + 64 - 1) / 64;
            try (FileChannel channel = FileChannel.open(file)) {
                if (channel.size() != fileSize) {
                    throw new GpJSONException("Size of file has changed while reading");
                }
                fileBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
                fileBuffer.load();
                UnsafeHelper.ByteArray byteArray = UnsafeHelper.createByteArray(fileBuffer);
                fileMemory.invokeMember("copyFrom", byteArray.getAddress());
            } catch (IOException e) {
                throw new GpJSONException("Failed to open file");
            }
            LOGGER.log(Level.FINER, "loadFile() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            isLoaded = true;
        }
    }

    private void buildIndexes(long numLevels) {
        if (!isIndexed || numLevels > this.numLevels) {
            if (!isLoaded)
                throw new GpJSONException("You must load the file before indexing");
            this.numLevels = numLevels;
            long start;
            start = System.nanoTime();
            this.createNewlineStringIndex();
            LOGGER.log(Level.FINER, "createNewlineStringIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            start = System.nanoTime();
            this.createLeveledBitmapsIndex();
            LOGGER.log(Level.FINER, "createLeveledBitmapsIndex() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
            isIndexed = true;
        }
    }

    private JSONPathResult compileQuery(String query) throws JSONPathException {
        long start = System.nanoTime();
        JSONPathResult result;
        result = new JSONPathParser(new JSONPathScanner(query)).compile();
        LOGGER.log(Level.FINER, "compileQuery() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return result;
    }

    protected abstract void createNewlineStringIndex();

    private void createLeveledBitmapsIndex() {
        leveledBitmapsIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize * numLevels);
        long startInitialize = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize).execute(leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), 0);
        LOGGER.log(Level.FINEST, "initialize done in " + (System.nanoTime() - startInitialize) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        Value carryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        kernels.get("create_leveled_bitmaps_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemory);
        int level = -1;
        for (int i=0; i<carryIndexMemory.getArraySize(); i++) {
            int value = carryIndexMemory.getArrayElement(i).asInt();
            carryIndexMemory.setArrayElement(i, level);
            level += value;
        }
        kernels.get("create_leveled_bitmaps").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringIndexMemory, carryIndexMemory, leveledBitmapsIndexMemory, levelSize * numLevels, levelSize, numLevels);
    }

    private long[][] query(JSONPathResult compiledQuery) {
        long start = System.nanoTime();
        if (!isIndexed)
            throw new GpJSONException("You must index the file before querying");
        long numberOfLines = newlineIndexMemory.getArraySize();
        long numberOfResults = compiledQuery.getNumResults();
        Value result = cu.invokeMember("DeviceArray", "long", numberOfLines * 2 * numberOfResults);
        Value queryMemory = cu.invokeMember("DeviceArray", "char", compiledQuery.getIr().size());
        long startInitialize = System.nanoTime();
        kernels.get("initialize").execute(gridSize, blockSize).execute(result, result.getArraySize(), -1);
        LOGGER.log(Level.FINEST, "initialize done in " + (System.nanoTime() - startInitialize) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        byte[] queryByteArray = compiledQuery.getIr().toByteArray();
        for (int j = 0; j < queryByteArray.length; j++) {
            queryMemory.setArrayElement(j, queryByteArray[j]);
        }
        kernels.get("find_value").execute(512, 1024).execute(fileMemory, fileMemory.getArraySize(), newlineIndexMemory, newlineIndexMemory.getArraySize(), stringIndexMemory, leveledBitmapsIndexMemory, leveledBitmapsIndexMemory.getArraySize(), levelSize, queryMemory, compiledQuery.getNumResults(), result);
        UnsafeHelper.LongArray longArray = UnsafeHelper.createLongArray(result.getArraySize());
        result.invokeMember("copyTo", longArray.getAddress());
        long[][] resultIndexes = new long[(int) numberOfLines][(int) numberOfResults * 2];
        for (int j = 0; j < numberOfLines; j++) {
            for (int k = 0; k < compiledQuery.getNumResults()*2; k+=2) {
                resultIndexes[j][k] = longArray.getValueAt(j*numberOfResults*2+ k);
                resultIndexes[j][k+1] = longArray.getValueAt(j*numberOfResults*2 + k + 1);
            }
        }
        LOGGER.log(Level.FINER, "query() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return resultIndexes;
    }

    private List<List<String>> fallbackQuery(String query) {
        Configuration conf = Configuration.defaultConfiguration()
                .addOptions(Option.ALWAYS_RETURN_LIST, Option.ALWAYS_RETURN_LIST);
        ParseContext parseContext = JsonPath.using(conf);

        Path file = Paths.get(fileName);
        JsonPath compiledQuery = JsonPath.compile(query);

        try (Stream<String> lines = Files.lines(file, StandardCharsets.UTF_8)) {
            return lines.parallel().map(line -> {
                List<String> result;
                try {
                    result = Collections.singletonList(parseContext.parse(line).read(compiledQuery).toString());
                } catch (PathNotFoundException e) {
                    result = Collections.emptyList();
                }
                return result;
            }).collect(Collectors.toList());
        } catch (IOException e) {
            throw new GpJSONException("Failed to read file");
        }
    }

    public Result query(String[] queries) {
        this.loadFile();
        JSONPathResult[] compiledQueries = new JSONPathResult[queries.length];
        int maxDepth = 0;
        for (int i=0; i< queries.length; i++) {
            try {
                compiledQueries[i] = this.compileQuery(queries[i]);
                maxDepth = Math.max(maxDepth, compiledQueries[i].getMaxDepth());
            } catch (UnsupportedJSONPathException e) {
                LOGGER.log(Level.FINE, "Unsupported JSONPath query '" + queries[i] + "'. Falling back to cpu execution");
                compiledQueries[i] = null;
            } catch (JSONPathException e) {
                throw new GpJSONException("Error parsing query: " + queries[i]);
            }
        }
        this.buildIndexes(maxDepth);
        Result result = new Result();
        for (int i = 0; i < compiledQueries.length; i++) {
            JSONPathResult compiledQuery = compiledQueries[i];
            String query = queries[i];
            if (compiledQuery != null) {
                result.addQuery(this.query(compiledQuery), this.fileBuffer);
                LOGGER.log(Level.FINE, query + " executed successfully");
            } else {
                result.addFallbackQuery(this.fallbackQuery(query));
                LOGGER.log(Level.FINE, query + " executed successfully (cpu fallback)");
            }
        }
        return result;
    }

    private ResultQuery query(String query) {
        JSONPathResult compiledQuery;
        try {
            compiledQuery = this.compileQuery(query);
        } catch (UnsupportedJSONPathException e) {
            LOGGER.log(Level.FINE, "Unsupported JSONPath query '" + query + "'. Falling back to cpu execution");
            compiledQuery = null;
        } catch (JSONPathException e) {
            throw new GpJSONException("Error parsing query: " + query);
        }

        ResultQuery result;
        if (compiledQuery != null) {
            long[][] values = this.query(compiledQuery);
            result = new ResultGPJSONQuery(values.length, values, fileBuffer);
            LOGGER.log(Level.FINE, query + " executed successfully");
        } else {
            result = new ResultFallbackQuery(this.fallbackQuery(query));
            LOGGER.log(Level.FINE, query + " executed successfully (cpu fallback)");
        }
        return result;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new String[] {LOADFILE, BUILDINDEXES, QUERY};
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return LOADFILE.equals(member) | BUILDINDEXES.equals(member) | QUERY.equals(member);
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments) throws UnknownIdentifierException, UnsupportedTypeException {
        switch (member) {
            case LOADFILE:
                if (arguments.length != 0) {
                    throw new GpJSONException(LOADFILE + " function requires 0 arguments");
                }
                this.loadFile();
                return this;
            case BUILDINDEXES:
                if (arguments.length != 1) {
                    throw new GpJSONException(BUILDINDEXES + " function requires 1 argument");
                }
                int depth = InvokeUtils.expectInt(arguments[0], "argument 1 of " + BUILDINDEXES + " must be an int");
                this.buildIndexes(depth);
                return this;
            case QUERY:
                if (arguments.length != 1) {
                    throw new GpJSONException(QUERY + " function requires 1 arguments");
                }
                String query = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                return this.query(query);
            default:
                throw UnknownIdentifierException.create(member);
        }
    }
}
