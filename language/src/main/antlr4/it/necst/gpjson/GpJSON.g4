grammar GpJSON ;

@parser::header
{
import java.util.ArrayList;
import java.util.Optional;
import it.necst.gpjson.nodes.ExpressionNode;
import it.necst.gpjson.nodes.EngineNode;
import it.necst.gpjson.parser.GpJSONParserException;
import it.necst.gpjson.parser.NodeFactory;
import it.necst.gpjson.GpJSONLanguage;
import com.oracle.truffle.api.source.Source;
}

@parser::members
{
private NodeFactory factory;

public static ExpressionNode parse(Source source) {
    GpJSONLexer lexer = new GpJSONLexer(CharStreams.fromString(source.getCharacters().toString()));
    GpJSONParser parser = new GpJSONParser(new CommonTokenStream(lexer));
    lexer.removeErrorListeners();
    parser.removeErrorListeners();
    parser.factory = new NodeFactory(source);
    ParserErrorListener parserErrorListener = new ParserErrorListener(source);
    parser.addErrorListener(parserErrorListener);
    ExpressionNode expression = parser.expr().result;
    Optional<GpJSONParserException> maybeException = parserErrorListener.getException();
    if (maybeException.isPresent()) {
      throw maybeException.get();
    } else {
      return expression;
    }
}

private static class ParserErrorListener extends BaseErrorListener {
    private GpJSONParserException exception;
    private Source source;

    ParserErrorListener(Source source) {
      this.source = source;
    }

    @Override
    public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                            int line, int charPositionInLine,
                            String msg, RecognitionException e) {
      Token token = (Token) offendingSymbol;
      exception = new GpJSONParserException(msg, source, line, charPositionInLine,
                                            Math.max(token.getStopIndex() - token.getStartIndex(), 0));
    }

    public Optional<GpJSONParserException> getException() {
      return Optional.ofNullable(exception);
    }
}
}

// parser
expr returns [ExpressionNode result]
  : engine EOF    { $result = $engine.result; }
  ;

engine returns [EngineNode result]
  :  'GJ'        { $result = factory.createEngineNode(); }
  ;

// lexer
String: '"' StringChar* '"';
Identifier: Letter (Letter | Digit)*;
IntegerLiteral: Digit+;

fragment Digit: [0-9];
fragment Letter: [A-Z] | [a-z] | '_' | '$';
fragment StringChar: ~('"' | '\\' | '\r' | '\n');

WS: (' ' | '\t')+ -> skip;
Comment: '/*' .*? '*/' -> skip;
LineComment: '//' ~[\r\n]* -> skip;
NL: '\r'? '\n' -> skip;
