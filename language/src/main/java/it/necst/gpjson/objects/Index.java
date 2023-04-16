package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.GpJSONOptionMap;
import it.necst.gpjson.engine.core.*;
import it.necst.gpjson.engine.disk.SavedIndex;
import it.necst.gpjson.engine.disk.SavedIndexBuilder;
import it.necst.gpjson.jsonpath.JSONPathQuery;
import it.necst.gpjson.utils.HashHelper;
import it.necst.gpjson.utils.InvokeUtils;
import org.graalvm.polyglot.Value;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

@ExportLibrary(InteropLibrary.class)
public class Index implements TruffleObject {
    private static final String QUERY = "query";
    private static final String FREE = "free";
    private static final String SAVE = "save";

    private static final Set<String> MEMBERS = new HashSet<>(Arrays.asList(QUERY, FREE, SAVE));

    private final Value cu;
    private final Map<String,Value> kernels;
    private final DataBuilder[] dataBuilder;
    private final IndexBuilder[] indexBuilder;
    private final int numPartitions;
    private boolean isFreed = false;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Index(Value cu, Map<String, Value> kernels, DataBuilder[] dataBuilder, IndexBuilder[] indexBuilder, int numPartitions) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.indexBuilder = indexBuilder;
        this.numPartitions = numPartitions;
    }

    public Index(Value cu, Map<String, Value> kernels, DataBuilder[] dataBuilder, SavedIndexBuilder[] savedIndexBuilders, int numPartitions) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.numPartitions = numPartitions;

        this.indexBuilder = new IndexBuilder[numPartitions];
        for (int i = 0; i < numPartitions; i++) {
            indexBuilder[i] = new IndexBuilder(cu, kernels, dataBuilder[i], savedIndexBuilders[i]);
        }
    }

    public void free() {
        for (int i=0; i < numPartitions; i++)
            indexBuilder[i].free();
        isFreed = true;
    }

    public Result query(String[] queries, JSONPathQuery[] compiledQueries) {
        if (isFreed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("You can't operate on a freed index");
        }
        Result result = new Result();
        for (int i=0; i < queries.length; i++) {
            result.addQuery(doQuery(queries[i], compiledQueries[i]));
        }
        return result;
    }

    private Result query(String query) {
        if (isFreed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("You can't operate on a freed index");
        }
        QueryCompiler queryCompiler = new QueryCompiler(new String[] {query});
        return query(new String[] {query}, queryCompiler.getCompiledQueries());
    }

    private void save(String fileName) {
        SavedIndexBuilder[] savedIndexBuilders = new SavedIndexBuilder[numPartitions];
        for (int i = 0; i < indexBuilder.length; i++) {
            savedIndexBuilders[i] = indexBuilder[i].save();
        }
        SavedIndex savedIndex = new SavedIndex(savedIndexBuilders, GpJSONOptionMap.getPartitionSize(), HashHelper.computeHash(dataBuilder, numPartitions));
        savedIndex.save(fileName);
    }

    private ResultQuery doQuery(String query, JSONPathQuery compiledQuery) {
        ResultQuery result;
        if (compiledQuery != null) {
            if (compiledQuery.getMaxDepth() > indexBuilder[0].getNumLevels())
                throw new GpJSONException("Query " + query + "requires " + compiledQuery.getMaxDepth() + "levels, but index has " + indexBuilder[0].getNumLevels());
            result = doGPJSONQuery(compiledQuery);
            LOGGER.log(Level.FINE, query + " executed successfully");
        } else {
            result = new ResultFallbackQuery(FallbackQueryExecutor.fallbackQuery(dataBuilder[0].getFileName(), query));
            LOGGER.log(Level.FINE, query + " executed successfully (cpu fallback)");
        }
        return result;
    }

    private ResultQuery doGPJSONQuery(JSONPathQuery compiledQuery) {
        ResultGPJSONQuery result;
        result = new ResultGPJSONQuery();
        QueryExecutor[] queryExecutor = new QueryExecutor[numPartitions];
        int stride = GpJSONOptionMap.getStride();
        for (int s=0; s < numPartitions/stride + 1; s++) {
            int strideStart = s * stride;
            int strideEnd = Math.min((s+1) * stride, numPartitions);
            for (int i=strideStart; i < strideEnd; i++) {
                indexBuilder[i].intermediateFree();
                queryExecutor[i] = new QueryExecutor(cu, kernels, dataBuilder[i], indexBuilder[i], compiledQuery);
                LOGGER.log(Level.FINER,  "Partition: " + i + ": queried successfully");
            }
            for (int i=strideStart; i < strideEnd; i++) {
                result.addPartition(queryExecutor[i].copyBuildResultArray(), dataBuilder[i].getFileBuffer(), indexBuilder[i].getNumLines());
                queryExecutor[i].free();
                LOGGER.log(Level.FINER,  "Partition: " + i + ": results fetched successfully");
            }
        }
        return result;
    }

    private boolean isFreed() {
        return isFreed;
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
    public Object invokeMember(String member, Object[] arguments) throws UnknownIdentifierException, UnsupportedTypeException, ArityException {
        switch (member) {
            case QUERY:
                if ((arguments.length != 1)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(1, 1, arguments.length);
                }
                String query = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                return query(query);
            case FREE:
                if ((arguments.length != 0)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(0, 0, arguments.length);
                }
                free();
                return this;
            case SAVE:
                if ((arguments.length != 1)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(1, 1, arguments.length);
                }
                String fileName = InvokeUtils.expectString(arguments[0], "argument 1 of " + SAVE + " must be a string");
                save(fileName);
                return this;
            default:
                CompilerDirectives.transferToInterpreter();
                throw UnknownIdentifierException.create(member);
        }
    }
}
