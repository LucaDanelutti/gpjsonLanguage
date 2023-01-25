package it.necst.gpjson;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.TruffleLanguage;
import it.necst.gpjson.nodes.ExpressionNode;
import it.necst.gpjson.nodes.GpJSONRootNode;
import it.necst.gpjson.parser.ParserAntlr;

@TruffleLanguage.Registration(id = "gpjson", name = "gpjson")
public final class GpJSONLanguage extends TruffleLanguage<Void> {
    protected CallTarget parse(ParsingRequest request) throws Exception {
        ExpressionNode exprNode = new ParserAntlr().parse(request.getSource());
        GpJSONRootNode rootNode = new GpJSONRootNode(this, exprNode);
        return rootNode.getCallTarget();
    }

    @Override
    protected Void createContext(Env env) {
        return null;
    }
}

