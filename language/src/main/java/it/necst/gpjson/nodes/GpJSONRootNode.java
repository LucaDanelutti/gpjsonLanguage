package it.necst.gpjson.nodes;

import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.RootNode;
import it.necst.gpjson.GpJSONLanguage;

public class GpJSONRootNode extends RootNode {
    @Child private final ExpressionNode expression;

    public GpJSONRootNode(GpJSONLanguage language, ExpressionNode expressionNode) {
        super(language);
        this.expression = expressionNode;
    }

    @Override
    public Object execute(VirtualFrame frame) {
        return expression.execute(frame);
    }
}
