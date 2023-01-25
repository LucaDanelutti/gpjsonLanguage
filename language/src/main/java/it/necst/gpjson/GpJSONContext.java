package it.necst.gpjson;

import com.oracle.truffle.api.TruffleLanguage;

public final class GpJSONContext {
    private final TruffleLanguage.Env env;

    public GpJSONContext(TruffleLanguage.Env env) {
        this.env = env;
    }
}
