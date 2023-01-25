package it.necst.gpjson.parser;

import com.oracle.truffle.api.CompilerAsserts;
import com.oracle.truffle.api.source.Source;
import it.necst.gpjson.GpJSONParser;
import it.necst.gpjson.nodes.ExpressionNode;

public final class ParserAntlr {

    public ParserAntlr() {
    }

    @SuppressWarnings("static-method")
    public ExpressionNode parse(Source source) throws GpJSONParserException {
        CompilerAsserts.neverPartOfCompilation();
        return GpJSONParser.parse(source);
    }
}
