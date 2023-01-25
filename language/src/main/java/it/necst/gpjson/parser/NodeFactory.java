package it.necst.gpjson.parser;

import com.oracle.truffle.api.source.Source;
import it.necst.gpjson.nodes.NewObjectNode;
import it.necst.gpjson.nodes.NewObjectNodeGen;
import org.antlr.v4.runtime.Token;

public class NodeFactory {

    private final Source source;

    public NodeFactory(Source source) {
        this.source = source;
    }

    public NewObjectNode createNewObjectNode(Token identifierToken) {
        return NewObjectNodeGen.create(identifierToken.getText());
    }
}
