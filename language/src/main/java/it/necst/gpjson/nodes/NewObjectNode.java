package it.necst.gpjson.nodes;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Specialization;
import it.necst.gpjson.Engine;
import it.necst.gpjson.GpJSONException;

public abstract class NewObjectNode extends ExpressionNode {
    private final String className;

    public NewObjectNode(String className) {
        this.className = className;
    }

    public String getClassName() {
        return className;
    }

    @Specialization
    protected Object doDefault() {
        switch (className) {
            case "Engine":
                return new Engine();
        default:
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONException("Invalid class '" + className + "'");
        }
    }
}
