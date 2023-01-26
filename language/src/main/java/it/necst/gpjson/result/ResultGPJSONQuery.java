package it.necst.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.MappedByteBuffer;

@ExportLibrary(InteropLibrary.class)
public class ResultGPJSONQuery extends ResultQuery implements TruffleObject {
    private final long numberOfLines;
    private final long[][] lines;
    private final MappedByteBuffer file;

    public ResultGPJSONQuery(long numberOfLines, long[][] lines, MappedByteBuffer file) {
        this.numberOfLines = numberOfLines;
        this.lines = lines;
        this.file = file;
    }

    public long[] getLine(int index) {
        return this.lines[index];
    }

    public MappedByteBuffer getFile() {
        return this.file;
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
        if (index >= numberOfLines) {
            throw InvalidArrayIndexException.create(index);
        }

        return new ResultGPJSONLine(this, index, this.lines[(int) index].length / 2);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.numberOfLines;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.numberOfLines;
    }
}
