package it.necst.gpjson;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.TruffleLanguage;
import it.necst.gpjson.nodes.ExpressionNode;
import it.necst.gpjson.nodes.GpJSONRootNode;
import it.necst.gpjson.parser.ParserAntlr;
import org.graalvm.options.OptionDescriptors;

@TruffleLanguage.Registration(id = GpJSONLanguage.ID, name = "gpjson")
public final class GpJSONLanguage extends TruffleLanguage<GpJSONContext> {
    public static final String ID = "gpjson";

    protected CallTarget parse(ParsingRequest request) {
        ExpressionNode exprNode = new ParserAntlr().parse(request.getSource());
        GpJSONRootNode rootNode = new GpJSONRootNode(this, exprNode);
        return rootNode.getCallTarget();
    }

    @Override
    protected GpJSONContext createContext(Env env) {
        return new GpJSONContext(env);
    }

    @Override
    public OptionDescriptors getOptionDescriptors() {
        return OptionDescriptors.createUnion(new GpJSONOptionsOptionDescriptors(), new GrCUDAOptionsOptionDescriptors());
    }
}

