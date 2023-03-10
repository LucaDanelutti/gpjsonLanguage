package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.GpJSONException;
import it.necst.gpjson.GpJSONLogger;
import it.necst.gpjson.engine.core.DataBuilder;
import it.necst.gpjson.engine.core.IndexBuilder;
import it.necst.gpjson.utils.InvokeUtils;
import org.graalvm.polyglot.Value;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static it.necst.gpjson.GpJSONLogger.GPJSON_LOGGER;

@ExportLibrary(InteropLibrary.class)
public class File implements TruffleObject {
    private static final String INDEX = "index";
    private static final String FREE = "free";

    private static final Set<String> MEMBERS = new HashSet<>(Arrays.asList(INDEX, FREE));

    private final Value cu;
    private final Map<String,Value> kernels;
    private final DataBuilder[] dataBuilder;
    private final int numPartitions;
    private boolean isFreed = false;

    private static final TruffleLogger LOGGER = GpJSONLogger.getLogger(GPJSON_LOGGER);

    public File(Value cu, Map<String, Value> kernels, DataBuilder[] dataBuilder, int numPartitions) {
        this.cu = cu;
        this.kernels = kernels;
        this.dataBuilder = dataBuilder;
        this.numPartitions = numPartitions;
    }

    public void free() {
        for (int i=0; i < numPartitions; i++)
            dataBuilder[i].free();
        isFreed = true;
    }

    public Index index(int depth, boolean combined) {
        if (isFreed()) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("You can't operate on a freed file");
        }
        IndexBuilder[] indexBuilder = new IndexBuilder[numPartitions];
        for (int i=0; i < numPartitions; i++) {
            indexBuilder[i] = new IndexBuilder(cu, kernels, dataBuilder[i], combined, depth);
        }
        return new Index(cu, kernels, dataBuilder, indexBuilder, numPartitions);
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
            case INDEX:
                if ((arguments.length != 2)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(2, 2, arguments.length);
                }
                int depth = InvokeUtils.expectInt(arguments[0], "argument 1 of " + INDEX + " must be an int");
                boolean combined = InvokeUtils.expectBoolean(arguments[1], "argument 2 of " + INDEX + " must be a boolean");
                return index(depth, combined);
            case FREE:
                if ((arguments.length != 0)) {
                    CompilerDirectives.transferToInterpreter();
                    throw ArityException.create(0, 0, arguments.length);
                }
                free();
                return this;
            default:
                CompilerDirectives.transferToInterpreter();
                throw UnknownIdentifierException.create(member);
        }
    }
}
