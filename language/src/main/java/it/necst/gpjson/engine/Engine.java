package it.necst.gpjson.engine;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.*;
import it.necst.gpjson.jsonpath.UnsupportedJSONPathException;
import it.necst.gpjson.kernel.GpJSONKernel;
import it.necst.gpjson.result.Result;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

@ExportLibrary(InteropLibrary.class)
public class Engine implements TruffleObject {
    private final Value cu;
    Map<String,Value> kernels = new HashMap<>();

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Engine() {
        final Context polyglot = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("grcuda.ExecutionPolicy", "async")
                // logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "FINER")
                .build();
        LOGGER.log(Level.FINE, "grcuda context created");
        cu = polyglot.eval("grcuda", "CU");
        LOGGER.log(Level.FINE, "Engine created");
    }

    public void buildKernels() {
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

    public Result query(String fileName, String[] queries, boolean combined) {
        if (kernels.isEmpty()) buildKernels();
        ExecutionContext exContext;

        if (combined)
            exContext = new ExecutionContextCombined(cu, kernels, fileName);
        else
            exContext = new ExecutionContextUncombined(cu, kernels, fileName);

        return exContext.execute(queries);
    }

    public void query(String filename, String query, boolean combined, int numLevels) {
        String[] queries = new String[1];
        queries[0] = query;
        this.query(filename, queries, combined);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new String[] {"buildKernels", "query"};
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return "buildKernels".equals(member) | "query".equals(member);
    }

    @ExportMessage
    public Object invokeMember(String member, Object[] arguments) throws UnknownIdentifierException, UnsupportedTypeException {
        switch (member) {
            case "buildKernels":
                if (arguments.length != 0) {
                    throw new GpJSONException("buildKernels function requires 0 arguments");
                }
                this.buildKernels();
                return this;
            case "query":
                if ((arguments.length != 3) && (arguments.length != 4)) {
                    throw new GpJSONException("query function requires 3 or 4 arguments");
                }
                String file = InvokeUtils.expectString(arguments[0], "argument 1 of query must be a string");
                String[] queries = InvokeUtils.expectStringArray(arguments[1], "argument 2 of query must be an array of strings");
                boolean combined = InvokeUtils.expectBoolean(arguments[2], "argument 3 of query must be a boolean");
                int numLevels = InvokeUtils.expectInt(arguments[3], "argument 3 of query must be an int");
                return this.query(file, queries, combined);
            default:
                throw UnknownIdentifierException.create(member);
        }
    }
}
