package it.necst.gpjson.nodes;

import com.oracle.truffle.api.dsl.Specialization;
import it.necst.gpjson.Engine;

public abstract class TestNode extends ExpressionNode {

    private final String identifierName;

    public TestNode(String identifierName) {
        this.identifierName = identifierName;
    }

    public String getIdentifierName() {
        return identifierName;
    }

    @Specialization
    protected Engine doDefault() {
            return new Engine();
    }
}
