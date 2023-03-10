package it.necst.gpjson.engine;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.GpJSONOptionMap;
import it.necst.gpjson.engine.core.DataLoader;
import it.necst.gpjson.engine.core.QueryCompiler;
import it.necst.gpjson.jsonpath.JSONPathQuery;
import it.necst.gpjson.kernel.GpJSONKernel;
import it.necst.gpjson.objects.File;
import it.necst.gpjson.objects.Index;
import it.necst.gpjson.utils.InvokeUtils;
import it.necst.gpjson.objects.Result;
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
    private static final String LOAD = "load";
    private static final String CLOSE = "close";

    private static final Set<String> MEMBERS = new HashSet<>(Arrays.asList(BUILDKERNELS, QUERY, LOAD, CLOSE));

    private final Context polyglot;
    private final Value cu;
    Map<String,Value> kernels = new HashMap<>();

    private final int partitionSize = GpJSONOptionMap.getPartitionSize();

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public Engine(Map<String,String> grCUDAOptions) {
        polyglot = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .options(grCUDAOptions)
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
                        if (inputStream.read(targetArray) <= 0) {
                            CompilerDirectives.transferToInterpreter();
                            throw new GpJSONInternalException("error reading from " + kernel.getFilename());
                        }
                        code = new String(targetArray, StandardCharsets.UTF_8);
                        kernels.put(kernel.getName(), cu.invokeMember("buildkernel", code, kernel.getParameterSignature()));
                    } else {
                        CompilerDirectives.transferToInterpreter();
                        throw new GpJSONInternalException("file not found " + kernel.getFilename());
                    }
                } catch (IOException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new GpJSONInternalException("cannot read from " + kernel.getFilename());
                }
            }
            LOGGER.log(Level.FINER, "buildKernels() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        }
    }

    private Result query(String fileName, String[] queries, boolean combined, boolean batched) {
        File file = load(fileName, batched);
        QueryCompiler queryCompiler = new QueryCompiler(queries);
        JSONPathQuery[] compiledQueries = queryCompiler.getCompiledQueries();
        Index index = file.index(queryCompiler.getMaxDepth(), combined);
        Result result = index.query(queries, compiledQueries);
        index.free();
        file.free();
        return result;
    }

    private File load(String fileName, boolean batched) {
        if (kernels.isEmpty()) buildKernels();
        DataLoader dataLoader;
        if (batched) {
            dataLoader = new DataLoader(cu, kernels, fileName, partitionSize);
        } else {
            dataLoader = new DataLoader(cu, kernels, fileName, 0);
        }
        return new File(cu, kernels, dataLoader.getDataBuilder(), dataLoader.getNumPartitions());
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
            case BUILDKERNELS:
                if (arguments.length != 0) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(0, 0, arguments.length);
                }
                this.buildKernels();
                return this;
            case QUERY: {
                if ((arguments.length != 4)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(4, 4, arguments.length);
                }
                String file = InvokeUtils.expectString(arguments[0], "argument 1 of " + QUERY + " must be a string");
                String[] queries = InvokeUtils.expectStringArray(arguments[1], "argument 2 of " + QUERY + " must be an array of strings");
                boolean combined = InvokeUtils.expectBoolean(arguments[2], "argument 3 of " + QUERY + " must be a boolean");
                boolean batched = InvokeUtils.expectBoolean(arguments[3], "argument 4 of " + QUERY + " must be a boolean");
                return this.query(file, queries, combined, batched);
            }
            case LOAD:
                if ((arguments.length != 2)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(2, 2, arguments.length);
                }
                String fileName = InvokeUtils.expectString(arguments[0], "argument 1 of " + LOAD + " must be a string");
                boolean batched = InvokeUtils.expectBoolean(arguments[1], "argument 2 of " + LOAD + " must be a boolean");
                return this.load(fileName, batched);
            case CLOSE:
                if ((arguments.length != 0)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(0, 0, arguments.length);
                }
                polyglot.close();
                return this;
            default:
                CompilerDirectives.transferToInterpreter();
                throw UnknownIdentifierException.create(member);
        }
    }
}
