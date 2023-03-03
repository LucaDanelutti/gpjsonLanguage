package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.List;

@ExportLibrary(InteropLibrary.class)
public class ResultFallbackLine implements TruffleObject {
    private final List<String> value;

    public ResultFallbackLine(List<String> value) {
        this.value = value;
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
        if (index >= this.value.size()) {
            throw InvalidArrayIndexException.create(index);
        }

        return this.value.get((int) index);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.value.size();
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.value.size();
    }
}
