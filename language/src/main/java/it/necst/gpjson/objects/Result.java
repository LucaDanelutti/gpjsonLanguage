package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.ArrayList;
import java.util.List;

@ExportLibrary(InteropLibrary.class)
public class Result implements TruffleObject {
    private final List<ResultQuery> resultQueries;

    public Result() {
        this.resultQueries = new ArrayList<>();
    }

    public void merge(Result result) {
        for (int i = 0; i < resultQueries.size(); i++) {
            ResultQuery resultQuery = resultQueries.get(i);
            resultQuery.addPartitions(result.resultQueries.get(i));
        }
    }

    public void addQuery(ResultQuery query) {
        resultQueries.add(query);
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
        if (index >= this.resultQueries.size()) {
            throw InvalidArrayIndexException.create(index);
        }

        return this.resultQueries.get((int) index);
    }

    @ExportMessage
    @SuppressWarnings("unused")
    @CompilerDirectives.TruffleBoundary
    public boolean isArrayElementReadable(long index) {
        return index < this.resultQueries.size();
    }

    @ExportMessage
    @SuppressWarnings("unused")
    public long getArraySize() {
        return this.resultQueries.size();
    }
}
