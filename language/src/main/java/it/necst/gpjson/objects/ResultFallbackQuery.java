package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.List;

@ExportLibrary(InteropLibrary.class)
public class ResultFallbackQuery extends ResultQuery implements TruffleObject {
    private final List<List<String>> values;

    public ResultFallbackQuery(List<List<String>> values) {
        this.values = values;
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
        if (index >= this.values.size()) {
            throw InvalidArrayIndexException.create(index);
        }

        return new ResultFallbackLine(this.values.get((int) index));
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.values.size();
    }

    @ExportMessage
    public long getArraySize() {
        return this.values.size();
    }
}
