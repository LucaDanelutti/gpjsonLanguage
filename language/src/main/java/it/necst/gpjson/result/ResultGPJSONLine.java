package it.necst.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.charset.StandardCharsets;

@ExportLibrary(InteropLibrary.class)
public class ResultGPJSONLine implements TruffleObject {
    private final ResultGPJSONQuery array;
    private final long lineIndex;
    private final int numResults;

    public ResultGPJSONLine(ResultGPJSONQuery array, long lineIndex, int numResults) {
        this.array = array;
        this.lineIndex = lineIndex;
        this.numResults = numResults;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public Object readArrayElement(long index) throws InvalidArrayIndexException {
        if (index >= this.numResults) {
            throw InvalidArrayIndexException.create(index);
        }

        long valueIndex = index * 2;

        int valueStart = (int) this.array.getLine((int) this.lineIndex)[(int) valueIndex];
        if (valueStart == -1) {
            return NullValue.INSTANCE;
        }
        int valueEnd = (int) this.array.getLine((int) this.lineIndex)[(int) (valueIndex + 1)];

        byte[] value = new byte[valueEnd - valueStart];
        array.getFile().position(valueStart);
        array.getFile().get(value);

        return new String(value, StandardCharsets.UTF_8);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.numResults;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.numResults;
    }
}
