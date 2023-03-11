package it.necst.gpjson;

import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

@ExportLibrary(InteropLibrary.class)
public class GpJSONOptionMap implements TruffleObject {
    private static final HashMap<String, Object> optionsMap = new HashMap<>();

    private static final HashMap<OptionKey<?>, String> optionNames = new HashMap<>();

    public GpJSONOptionMap(OptionValues options) {
        options.getDescriptors().forEach(o -> {
            optionsMap.put(o.getName(), options.get(o.getKey()));
            optionNames.put(o.getKey(), o.getName());
        });
    }

    private static Object getOptionValueFromOptionKey(OptionKey<?> optionKey) {
        return optionsMap.get(optionNames.get(optionKey));
    }

    public static int getPartitionSize() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.PartitionSize);
    }

    public static int getStride() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.PartitionSize);
    }

    public static int getNumberOfGPUs() {
        return (Integer) getOptionValueFromOptionKey(GrCUDAOptions.NumberOfGPUs);
    }

    public static int getIndexBuilderGridSize() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.IndexBuilderGridSize);
    }

    public static int getIndexBuilderBlockSize() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.IndexBuilderBlockSize);
    }

    public static int getQueryExecutorGridSize() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.QueryExecutorGridSize);
    }

    public static int getQueryExecutorBlockSize() {
        return (Integer) getOptionValueFromOptionKey(GpJSONOptions.QueryExecutorBlockSize);
    }

    // Implement InteropLibrary;
    @ExportMessage
    public final boolean hasHashEntries(){
        return true;
    }

    @ExportMessage
    public final Object readHashValue(Object key) throws UnknownKeyException, UnsupportedMessageException {
        Object value;
        if (key instanceof String){
            value = this.optionsMap.get(key);
        }
        else {
            throw UnsupportedMessageException.create();
        }
        if (value == null) throw UnknownKeyException.create(key);
        return value.toString();
    }

    @ExportMessage
    public final long getHashSize(){
        return optionsMap.size();
    }

    @ExportMessage
    public final boolean isHashEntryReadable(Object key) {
        return key instanceof String && this.optionsMap.containsKey(key);
    }

    @ExportMessage
    public Object getHashEntriesIterator() {
        return new EntriesIterator(optionsMap.entrySet().iterator());
    }

    @ExportLibrary(InteropLibrary.class)
    public static final class EntriesIterator implements TruffleObject {
        private final Iterator<Map.Entry<String, Object>> iterator;

        private EntriesIterator(Iterator<Map.Entry<String, Object>> iterator) {
            this.iterator = iterator;
        }

        @SuppressWarnings("static-method")
        @ExportMessage
        public boolean isIterator() {
            return true;
        }

        @ExportMessage
        public boolean hasIteratorNextElement() {
            try {
                return iterator.hasNext();
            } catch(NoSuchElementException e) {
                return false;
            }
        }

        @ExportMessage
        public GrCUDAOptionTuple getIteratorNextElement() throws StopIterationException {
            if (hasIteratorNextElement()) {
                Map.Entry<String,Object> entry = iterator.next();
                return new GrCUDAOptionTuple(entry.getKey(), entry.getValue().toString());
            } else {
                throw StopIterationException.create();
            }
        }
    }

    @ExportLibrary(InteropLibrary.class)
    public static class GrCUDAOptionTuple implements TruffleObject {

        private final int SIZE = 2;
        private final String[] entry = new String[SIZE];

        public GrCUDAOptionTuple(String key, String value) {
            entry[0] = key;
            entry[1] = value;
        }

        @ExportMessage
        static boolean hasArrayElements(GrCUDAOptionTuple tuple) {
            return true;
        }

        @ExportMessage
        public final boolean isArrayElementReadable(long index) {
            return index == 0 || index == 1;
        }

        @ExportMessage
        public final Object readArrayElement(long index) throws InvalidArrayIndexException {
            if (index == 0 || index == 1) {
                return entry[(int)index];
            }
            else {
                throw InvalidArrayIndexException.create(index);
            }
        }

        @ExportMessage
        public final long getArraySize() {
            return SIZE;
        }
    }
}
