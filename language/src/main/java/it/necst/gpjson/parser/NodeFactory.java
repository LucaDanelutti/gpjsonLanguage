package it.necst.gpjson.parser;

import com.oracle.truffle.api.source.Source;
import it.necst.gpjson.nodes.TestNode;
import it.necst.gpjson.nodes.TestNodeGen;
import org.antlr.v4.runtime.Token;

public class NodeFactory {

    private final Source source;

    public NodeFactory(Source source) {
        this.source = source;
    }

    public TestNode createTestNode(Token identifierToken) {
        return TestNodeGen.create(identifierToken.getText());
    }
}
