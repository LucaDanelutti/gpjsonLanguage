package it.necst.gpjson.parser;

import com.oracle.truffle.api.source.Source;
import it.necst.gpjson.nodes.EngineNode;
import it.necst.gpjson.nodes.EngineNodeGen;

public class NodeFactory {
    private final Source source;

    public NodeFactory(Source source) {
        this.source = source;
    }

    public EngineNode createEngineNode() {
        return EngineNodeGen.create();
    }
}
