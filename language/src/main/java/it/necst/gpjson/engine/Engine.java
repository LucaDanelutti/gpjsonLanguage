package it.necst.gpjson.engine;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.*;
import it.necst.gpjson.jsonpath.UnsupportedJSONPathException;
import it.necst.gpjson.kernel.GpJSONKernel;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

@ExportLibrary(InteropLibrary.class)
public class Engine implements TruffleObject {
    private final Value cu;
    Map<String,Value> kernels = new HashMap<>();

    public Engine() {
        MyLogger.setLevel(Level.FINER);
        final Context polyglot = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                //.option("grcuda.EnableComputationTimers", "true")
                .option("grcuda.ExecutionPolicy", "sync")
                // logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "FINER")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "INFO")
                .build();
        MyLogger.log(Level.FINE, "Engine", "()", "grcuda context created");
        cu = polyglot.eval("grcuda", "CU");
        MyLogger.log(Level.FINE, "Engine", "()", "Engine created");
    }

    public void buildKernels() {
        if (kernels.isEmpty()) {
            long start;
            start = System.nanoTime();
            for (GpJSONKernel kernel : GpJSONKernel.values()) {
                InputStream inputStream = getClass().getClassLoader().getResourceAsStream(kernel.getFilename());
                if (inputStream != null) {
                    String code;
                    try {
                        byte[] targetArray = new byte[inputStream.available()];
                        inputStream.read(targetArray);
                        code = new String(targetArray, StandardCharsets.UTF_8);
                    } catch (IOException e) {
                        throw new GpJSONInternalException("cannot read from " + kernel.getFilename());
                    }
                    kernels.put(kernel.getName(), cu.invokeMember("buildkernel", code, kernel.getParameterSignature()));
                } else {
                    throw new GpJSONInternalException("file not found " + kernel.getFilename());
                }
            }
            MyLogger.log(Level.FINER, "Engine", "buildKernels()", "buildKernels() done in " + (System.nanoTime() - start) / (double) TimeUnit.MILLISECONDS.toNanos(1) + "ms");
        }
    }

    public void query(String fileName, String[] queries, boolean combined, int numLevels, boolean getStrings) {
        if (kernels.isEmpty()) buildKernels();
        ExecutionContext exContext;

        if (combined)
            exContext = new ExecutionContextCombined(cu, kernels, fileName);
        else
            exContext = new ExecutionContextUncombined(cu, kernels, fileName);

        exContext.loadFile();
        exContext.buildIndexes(numLevels);

        for (String query: queries) {
            try {
                if (getStrings) {
                    List<List<String>> resultStrings = exContext.executeAndGetStrings(query);
                    MyLogger.log(Level.FINE, "Engine", "call()", query + " executed successfully with " + resultStrings.size() + " results");
                    if (resultStrings.size() < 50)
                        MyLogger.log(Level.FINER, "Engine", "call()", resultStrings.toString());
                } else {
                    exContext.execute(query);
                    MyLogger.log(Level.FINE, "Engine", "call()", query + " executed successfully");
                }
            } catch (UnsupportedJSONPathException e) {
                MyLogger.log(Level.FINE, "Engine", "call()", "Unsupported JSONPath query \'" + query + "\'. Falling back to cpu execution");
                FallbackExecutionContext fallbackExecutionContext= new FallbackExecutionContext(fileName);
                List<List<String>> resultStrings = fallbackExecutionContext.execute(query);
                MyLogger.log(Level.FINE, "Engine", "call()", query + " executed successfully (fallback to cpu) with " + resultStrings.size() + " results");
                if (resultStrings.size() < 50)
                    MyLogger.log(Level.FINER, "Engine", "call()", resultStrings.toString());
            }
        }
    }

    public void query(String filename, String query, boolean combined, int numLevels, boolean getStrings) {
        String[] queries = new String[1];
        queries[0] = query;
        this.query(filename, queries, combined, numLevels, getStrings);
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
                break;
            case "query":
                if ((arguments.length != 3) && (arguments.length != 4)) {
                    throw new GpJSONException("query function requires 3 or 4 arguments");
                }
                String file = InvokeUtils.expectString(arguments[0], "argument 1 of query must be a string");
                String[] queries = InvokeUtils.expectStringArray(arguments[1], "argument 2 of query must be an array of strings");
                boolean combined = InvokeUtils.expectBoolean(arguments[2], "argument 3 of query must be a boolean");
                int numLevels = InvokeUtils.expectInt(arguments[3], "argument 3 of query must be an int");
                this.query(file, queries, combined, numLevels, false);
                break;
            default:
                throw UnknownIdentifierException.create(member);
        }
        return this;
    }
}
