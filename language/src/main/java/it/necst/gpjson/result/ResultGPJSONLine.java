package it.necst.gpjson.result;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.nio.MappedByteBuffer;
import java.nio.charset.StandardCharsets;

@ExportLibrary(InteropLibrary.class)
public class ResultGPJSONLine implements TruffleObject {
    private final int[] records;
    private final MappedByteBuffer file;

    public ResultGPJSONLine(int[] records, MappedByteBuffer file) {
        this.records = records;
        this.file = file;
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
        if (index >= this.records.length / 2) {
            throw InvalidArrayIndexException.create(index);
        }

        int valueIndex = (int) index * 2;

        int valueStart = this.records[valueIndex];
        if (valueStart == -1) {
            return NullValue.INSTANCE;
        }
        int valueEnd = this.records[valueIndex + 1];

        byte[] value = new byte[valueEnd - valueStart];
        file.position(valueStart);
        file.get(value);

        return new String(value, StandardCharsets.UTF_8);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.records.length / 2;
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.records.length / 2;
    }
}
