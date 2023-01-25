package it.necst.gpjson.nodes;

import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.NodeInfo;
import it.necst.gpjson.GpJSONContext;

@NodeInfo(description = "Abstract base node for all expressions")
public abstract class ExpressionNode extends Node {
    public abstract Object execute(VirtualFrame frame);

    protected final GpJSONContext currentLanguageContext() {
        return GpJSONContext.get(this);
    }
}
