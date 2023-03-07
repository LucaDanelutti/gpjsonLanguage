package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.jsonpath.*;

import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class QueryCompiler {
    private final JSONPathQuery[] compiledQueries;
    private int maxDepth = 1;
    private int totalSize = 0;
    private int totalNumResults = 0;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public QueryCompiler(String[] queries) {
        compiledQueries = compile(queries);
    }

    public JSONPathQuery[] getCompiledQueries() {
        return compiledQueries;
    }

    public JSONPathQuery getCompiledQuery(int index) {
        return compiledQueries[index];
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public int getNumQueries() {
        return compiledQueries.length;
    }

    public int getTotalSize() {
        return totalSize;
    }

    public int getTotalNumResults() {
        return totalNumResults;
    }

    private JSONPathQuery[] compile(String[] queries) {
        JSONPathQuery[] compiledQueries = new JSONPathQuery[queries.length];
        for (int i=0; i< queries.length; i++) {
            try {
                compiledQueries[i] = new JSONPathParser(new JSONPathScanner(queries[i])).compile();
                maxDepth = Math.max(maxDepth, compiledQueries[i].getMaxDepth());
                totalSize += compiledQueries[i].getIr().size();
                totalNumResults += compiledQueries[i].getNumResults();
            } catch (UnsupportedJSONPathException e) {
                LOGGER.log(Level.FINE, "Unsupported JSONPath query '" + queries[i] + "'. Falling back to cpu execution.");
                compiledQueries[i] = null;
            } catch (JSONPathException e) {
                throw new GpJSONException("Error parsing query: " + queries[i]);
            }
        }
        return compiledQueries;
    }
}
