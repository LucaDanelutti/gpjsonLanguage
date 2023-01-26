package it.necst.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.MappedByteBuffer;

@ExportLibrary(InteropLibrary.class)
public class Result implements TruffleObject {
    private final int numberOfQueries;
    private final ResultQuery[] resultQueries;

    public Result(long[][][] values, MappedByteBuffer file) {
        this.numberOfQueries = values.length;
        resultQueries = new ResultQuery[numberOfQueries];
        for (int i=0; i<values.length; i++) {
            resultQueries[i] = new ResultQuery(values[i].length, values[i], file);
        }
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
        if (index >= numberOfQueries) {
            throw InvalidArrayIndexException.create(index);
        }

        return resultQueries[(int) index];
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.numberOfQueries;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.numberOfQueries;
    }
}
