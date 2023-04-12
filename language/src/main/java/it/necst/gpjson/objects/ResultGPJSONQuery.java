package it.necst.gpjson.objects;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import it.necst.gpjson.utils.Pair;

import java.nio.MappedByteBuffer;
import java.util.*;

@ExportLibrary(InteropLibrary.class)
public class ResultGPJSONQuery extends ResultQuery implements TruffleObject {
    private long numberOfLines = 0;
    private final NavigableMap<Long, Pair<int[][], MappedByteBuffer>> lines;

    public ResultGPJSONQuery() {
        this.lines = new TreeMap<>();
    }

    public void addPartition(int[][] lines, MappedByteBuffer file, long numberOfLines) {
        this.lines.put(this.numberOfLines, new Pair<>(lines, file));
        this.numberOfLines += numberOfLines;
    }

    public void addPartitions(ResultQuery resultQuery) {
        ResultGPJSONQuery resultGPJSONQuery = (ResultGPJSONQuery) resultQuery;
        long prev = 0;
        Long[] keySet = resultGPJSONQuery.lines.keySet().toArray(new Long[0]);
        for (int i=0; i < keySet.length; i++) {
            this.lines.put(this.numberOfLines, resultGPJSONQuery.lines.get(keySet[i]));
            long numberOfLines = (i == keySet.length-1) ? resultGPJSONQuery.numberOfLines : keySet[i+1];
            this.numberOfLines += numberOfLines-prev;
            prev = numberOfLines;
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
        if (index >= numberOfLines) {
            throw InvalidArrayIndexException.create(index);
        }

        long partitionStart = lines.floorKey(index);
        Pair<int[][], MappedByteBuffer> pair = lines.get(partitionStart);
        return new ResultGPJSONLine(pair.getKey()[(int) (index-partitionStart)], pair.getValue());
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
