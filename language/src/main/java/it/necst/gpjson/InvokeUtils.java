package it.necst.gpjson;

import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class InvokeUtils {
    private static final InteropLibrary INTEROP = InteropLibrary.getFactory().getUncached();

    public static String expectString(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asString(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        }
    }

    public static int expectInt(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asInt(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        }
    }

    public static boolean expectBoolean(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        try {
            return INTEROP.asBoolean(argument);
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        }
    }

    public static String[] expectStringArray(Object argument, String errorMessage) throws UnsupportedTypeException {
        CompilerAsserts.neverPartOfCompilation();
        String[] res;
        try {
            long size = INTEROP.getArraySize(argument);
            res = new String[(int) size];
            for (int i=0; i < size; i++) {
                res[i] = INTEROP.asString(INTEROP.readArrayElement(argument, i));
            }
        } catch (UnsupportedMessageException e) {
            throw UnsupportedTypeException.create(new Object[]{argument}, errorMessage);
        } catch (InvalidArrayIndexException e) {
            throw new GpJSONInternalException("unexpected behavior");
        }
        return res;
    }
}
