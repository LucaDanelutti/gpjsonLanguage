package it.necst.gpjson;

import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.nodes.Node;

public final class GpJSONContext {
    private final Engine engine;

    public GpJSONContext() {
        this.engine = new Engine();
    }

    public Engine getEngine() {
        return engine;
    }

    public static GpJSONContext get(Node node) {
        return TruffleLanguage.ContextReference.create(GpJSONLanguage.class).get(node);
    }
}
