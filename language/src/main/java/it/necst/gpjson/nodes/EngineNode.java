package it.necst.gpjson.nodes;

import com.oracle.truffle.api.dsl.Specialization;

public abstract class EngineNode extends ExpressionNode {
    @Specialization
    protected Object doDefault() {
        return this.currentLanguageContext().getEngine();
    }
}
