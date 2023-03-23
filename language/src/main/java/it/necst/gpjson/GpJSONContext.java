package it.necst.gpjson;

import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.nodes.Node;
import it.necst.gpjson.engine.Engine;
import org.graalvm.options.OptionValues;

import java.util.HashMap;
import java.util.Map;

public final class GpJSONContext {
    private final Engine engine;
    private final GpJSONOptionMap gpJSONOptionMap;

    public GpJSONContext(TruffleLanguage.Env env) {
        OptionValues options = env.getOptions();

        this.gpJSONOptionMap = new GpJSONOptionMap(options);

        Map<String, String> grCUDAOptions = new HashMap<>();
        options.getDescriptors().forEach(o -> {
            if (o.getName().startsWith("gpjson.grcuda"))
                grCUDAOptions.put(o.getName().split("gpjson.")[1], options.get(o.getKey()).toString());
        });

        this.engine = new Engine(grCUDAOptions);
    }

    public GpJSONOptionMap getGpJSONOptionMap() {
        return gpJSONOptionMap;
    }

    public Engine getEngine() {
        return engine;
    }

    public static GpJSONContext get(Node node) {
        return TruffleLanguage.ContextReference.create(GpJSONLanguage.class).get(node);
    }

    public void cleanup() {
        engine.cleanup();
    }
}
