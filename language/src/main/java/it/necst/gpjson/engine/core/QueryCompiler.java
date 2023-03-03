package it.necst.gpjson.engine.core;

import com.oracle.truffle.api.TruffleLogger;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.jsonpath.*;

import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

public class QueryCompiler {
    private final JSONPathQuery[] compiledQueries;
    private int maxDepth;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public QueryCompiler(String[] queries) {
        compiledQueries = compile(queries);
    }

    public JSONPathQuery[] getCompiledQueries() {
        return compiledQueries;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    private JSONPathQuery[] compile(String[] queries) {
        JSONPathQuery[] compiledQueries = new JSONPathQuery[queries.length];
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
        return compiledQueries;
    }
}
