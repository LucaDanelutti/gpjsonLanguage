package it.necst.gpjson;

import com.oracle.truffle.api.TruffleLogger;

public class GpJSONLogger {
    public static final String GPJSON_LOGGER = "it.necst.gpjson";

    public static TruffleLogger getLogger(String name) {
        return TruffleLogger.getLogger(GpJSONLanguage.ID, name);
    }
}
