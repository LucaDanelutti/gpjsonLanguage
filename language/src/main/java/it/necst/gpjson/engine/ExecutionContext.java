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
public class ExecutionContext implements TruffleObject {
    private static final String LOADFILE = "loadFile";
    private static final String BUILDINDEXES = "buildIndexes";
    private static final String QUERY = "query";

    protected final Value cu;
    protected final Map<String,Value> kernels;

    private Executor executor;

    //File
    private final String fileName;
    private MappedByteBuffer fileBuffer;
    protected Value fileMemory;
    protected long levelSize;
    private boolean isLoaded = false;

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

    private void buildIndexes(long numLevels, boolean combined) {
        if (!isLoaded)
            throw new GpJSONException("You must load the file before indexing");
        this.executor = new Executor(cu, kernels, fileMemory, combined);
        executor.buildIndexes(numLevels);
    }

    private JSONPathResult compileQuery(String query) throws JSONPathException {
        long start = System.nanoTime();
        JSONPathResult result;
        result = new JSONPathParser(new JSONPathScanner(query)).compile();
        LOGGER.log(Level.FINER, "compileQuery() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        return result;
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

    public Result query(String[] queries, boolean combined) {
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
        this.buildIndexes(maxDepth, combined);
        Result result = new Result();
        for (int i = 0; i < compiledQueries.length; i++) {
            JSONPathResult compiledQuery = compiledQueries[i];
            String query = queries[i];
            if (compiledQuery != null) {
                result.addQuery(this.executor.query(compiledQuery), this.fileBuffer);
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
            long[][] values = this.executor.query(compiledQuery);
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
                if (arguments.length != 2) {
                    throw new GpJSONException(BUILDINDEXES + " function requires 2 argument");
                }
                int depth = InvokeUtils.expectInt(arguments[0], "argument 1 of " + BUILDINDEXES + " must be an int");
                boolean combined = InvokeUtils.expectBoolean(arguments[1], "argument 2 of " + BUILDINDEXES + " must be a boolean");
                this.buildIndexes(depth, combined);
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
