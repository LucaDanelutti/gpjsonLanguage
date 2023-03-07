package it.necst.gpjson.objects;

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
import it.necst.gpjson.engine.core.FallbackQueryExecutor;
import it.necst.gpjson.engine.core.DataBuilder;
import it.necst.gpjson.engine.core.IndexBuilder;
import it.necst.gpjson.engine.core.QueryCompiler;
import it.necst.gpjson.engine.core.QueryExecutor;
import it.necst.gpjson.jsonpath.JSONPathQuery;
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

    private static final Set<String> MEMBERS = new HashSet<>(Arrays.asList(QUERY, FREE));

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

    public void free() {
        for (int i=0; i < numPartitions; i++)
            indexBuilder[i].free();
        isFreed = true;
    }

    public Result query(String[] queries, JSONPathQuery[] compiledQueries, QueryCompiler queryCompiler) {
        if (isFreed())
            throw new GpJSONException("You can't operate on a freed index");
        return getResult(queries[0], compiledQueries[0], queryCompiler);
    }

    private Result query(String query) {
        if (isFreed())
            throw new GpJSONException("You can't operate on a freed index");
        QueryCompiler queryCompiler = new QueryCompiler(new String[] {query});
        JSONPathQuery compiledQuery = queryCompiler.getCompiledQueries()[0];
        return getResult(query, compiledQuery, queryCompiler);
    }

    private Result getResult(String query, JSONPathQuery compiledQuery, QueryCompiler queryCompiler) {
        Result result = new Result();
        if (compiledQuery != null) {
            QueryExecutor[] queryExecutor = new QueryExecutor[numPartitions];
            for (int i=0; i < numPartitions; i++) {
                queryExecutor[i] = new QueryExecutor(cu, kernels, dataBuilder[i], indexBuilder[i], queryCompiler);
            }

            for (int i=0; i < numPartitions; i++) {
                int[][][] indexes = queryExecutor[i].copyBuildResultArray();
                for (int[][] index : indexes) {
                    ResultGPJSONQuery resultQuery = new ResultGPJSONQuery();
                    resultQuery.addPartition(index, dataBuilder[i].getFileBuffer(), indexBuilder[i].getNumLines());
                    result.addQuery(resultQuery);
                }
                queryExecutor[i].free();
            }
            LOGGER.log(Level.FINE, query + " executed successfully");
        } else {
            ResultFallbackQuery resultQuery = new ResultFallbackQuery(FallbackQueryExecutor.fallbackQuery(dataBuilder[0].getFileName(), query));
            result.addQuery(resultQuery);
            LOGGER.log(Level.FINE, query + " executed successfully (cpu fallback)");
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
    public Object invokeMember(String member, Object[] arguments) throws UnknownIdentifierException, UnsupportedTypeException {
        switch (member) {
            case QUERY:
                if ((arguments.length != 1)) {
                    throw new GpJSONException(QUERY + " function requires 1 arguments");
                }
                String query = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                return query(query);
            case FREE:
                free();
                return this;
            default:
                throw UnknownIdentifierException.create(member);
        }
    }
}
