package it.necst.gpjson.engine;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.*;
import it.necst.gpjson.kernel.GpJSONKernel;
import it.necst.gpjson.result.Result;
import it.necst.gpjson.result.ResultGPJSONQuery;
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
    private static final String BUILDKERNELS = "buildKernels";
    private static final String QUERY = "query";
    private static final String CREATECONTEXT = "createContext";

    private final Value cu;
    Map<String,Value> kernels = new HashMap<>();

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Engine() {
        final Context polyglot = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("grcuda.ExecutionPolicy", "sync")
                .option("grcuda.DeviceSelectionPolicy", "min-transfer-size")
                .option("grcuda.NumberOfGPUs", "1")
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

    private Result query(String fileName, String[] queries, boolean combined, boolean batched) {
        if (kernels.isEmpty()) buildKernels();
        if (batched) {
            ResultGPJSONQuery query = new BatchedExecutionContext(cu, kernels, fileName, (int) Math.pow(2, 30)).query(queries[0], combined);
            Result result = new Result();
            result.addQuery(query);
            return result;
        } else
            return new ExecutionContext(cu, kernels, fileName).query(queries, combined);
    }

    private ExecutionContext createContext(String fileName) {
        if (kernels.isEmpty()) buildKernels();
        return new ExecutionContext(cu, kernels, fileName);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return new String[] {BUILDKERNELS, QUERY, CREATECONTEXT};
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberInvocable(String member) {
        return BUILDKERNELS.equals(member) | QUERY.equals(member) | CREATECONTEXT.equals(member);
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
            case QUERY:
                if ((arguments.length != 4)) {
                    throw new GpJSONException(QUERY + " function requires 4 arguments");
                }
                String file = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                String[] queries = InvokeUtils.expectStringArray(arguments[1], "argument 2 of " + QUERY + " must be an array of strings");
                boolean combined = InvokeUtils.expectBoolean(arguments[2], "argument 3 of " + QUERY + " must be a boolean");
                boolean batched = InvokeUtils.expectBoolean(arguments[3], "argument 4 of " + QUERY + " must be a boolean");
                return this.query(file, queries, combined, batched);
            case CREATECONTEXT:
                if ((arguments.length != 1)) {
                    throw new GpJSONException(CREATECONTEXT + " function requires 1 arguments");
                }
                file = InvokeUtils.expectString(arguments[0], "argument 1 of " + CREATECONTEXT + " must be a string");
                return this.createContext(file);
            default:
                throw UnknownIdentifierException.create(member);
        }
    }
}
