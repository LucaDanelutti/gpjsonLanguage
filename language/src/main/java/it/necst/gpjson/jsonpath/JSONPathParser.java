package it.necst.gpjson.jsonpath;

import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;

public class JSONPathParser {
    private final JSONPathScanner scanner;

    private final IRByteOutputBuffer output = new IRByteOutputBuffer();
    private final IRBuilder ir = new IRBuilder(output);
    private int maxLevel = 0;

    public JSONPathParser(JSONPathScanner scanner) {
        this.scanner = scanner;
    }

    public JSONPathParser(String string) {
        this(new JSONPathScanner(string));
    }

    public JSONPathQuery compile() throws JSONPathException {
        scanner.expectChar('$');
        compileNextExpression();
        ir.storeResult();
        ir.end();
        return new JSONPathQuery(output, maxLevel, ir.getNumResultStores());
    }

    private void compileNextExpression() throws JSONPathException {
        char c = scanner.peek();
        switch (c) {
            case '.':
                compileDotExpression();
                break;
            case '[':
                compileIndexExpression();
                break;
            default:
                throw scanner.unsupportedNext("Unsupported expression type");
        }
    }

    private void compileDotExpression() throws JSONPathException {
        scanner.expectChar('.');
        if (scanner.peek() == '.') {
            throw scanner.unsupportedNext("Unsupported recursive descent");
        }
        String property = readProperty();
        if (property.isEmpty()) {
            throw scanner.error("Unexpected empty property");
        }
        createPropertyIR(property);
        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void compileIndexExpression() throws JSONPathException {
        scanner.expectChar('[');
        if (scanner.peek() == '\'' || scanner.peek() == '"') {
            String property = readQuotedString();
            if (property.isEmpty()) {
                throw scanner.error("Unexpected empty property");
            }
            createPropertyIR(property);
        } else if (scanner.peek() >= '0' && scanner.peek() <= '9') {
            int index = readInteger(c -> c == ']' || c == ':' || c == ',');
            switch (scanner.peek()) {
                case ':':
                    scanner.expectChar(':');
                    int endIndex = readInteger(c -> c == ']');
                    scanner.expectChar(']');
                    compileIndexRangeExpression(index, endIndex, false);
                    return;
                case ',':
                    List<Integer> indexes = new ArrayList<>();
                    indexes.add(index);
                    while (scanner.peek() == ',') {
                        scanner.expectChar(',');
                        scanner.testDigit();
                        indexes.add(readInteger(c -> c == ',' || c == ']'));
                    }
                    scanner.expectChar(']');
                    compileMultipleIndexExpression(indexes);
                    return;
                case ']':
                    createIndexIR(index);
                    break;
            }
        } else if (scanner.peek() == ':') {
            scanner.expectChar(':');
            if (scanner.peek() >= '0' && scanner.peek() <= '9') {
                int index = readInteger(c -> c == ']');
                scanner.expectChar(']');
                compileIndexRangeExpression(0, index, false);
                return;
            } else {
                throw scanner.errorNext("Unexpected character in index, expected an integer");
            }
        } else if (scanner.peek() == '-') {
            scanner.expectChar('-');
            if (scanner.peek() >= '0' && scanner.peek() <= '9') {
                int index = readInteger(c -> c == ']' || c == ':');
                if (index == 0)
                    throw scanner.error("Invalid reverse index 0");
                if (scanner.peek() == ']')
                    createReverseIndexIR(index);
                else {
                    scanner.expectChar(':');
                    scanner.expectChar(']');
                    compileIndexRangeExpression(0, index, true);
                    return;
                }
            } else if (scanner.peek() == '-') {
                throw scanner.unsupportedNext("Unsupported last n elements of the array query");
            } else {
                throw scanner.errorNext("Unexpected character in index, expected an integer");
            }
        } else if (scanner.peek() == '*') {
            throw scanner.unsupportedNext("Unsupported wildcard expression");
        } else if (scanner.peek() == '?') {
            compileFilterExpression();
        } else {
            throw scanner.errorNext("Unexpected character in index, expected ', \", or an integer");
        }
        scanner.expectChar(']');
        if (scanner.hasNext()) {
            compileNextExpression();
        }
    }

    private void compileMultipleIndexExpression(List<Integer> indexes) throws JSONPathException {
        int maxMaxLevel = maxLevel;
        for (int j = 0; j < indexes.size(); j++) {
            Integer i = indexes.get(j);
            maxMaxLevel = compileIndexAux(i, j == indexes.size()-1, maxMaxLevel, false);
        }
        maxLevel = maxMaxLevel + 1;
    }

    private void compileIndexRangeExpression(int startIndex, int endIndex, boolean reverse) throws JSONPathException {
        int maxMaxLevel = maxLevel;
        if (reverse) {
            for (int i = endIndex; i > startIndex; i--) {
                maxMaxLevel = compileIndexAux(i, i == startIndex+1, maxMaxLevel, true);
            }
        } else {
            for (int i = startIndex; i < endIndex; i++) {
                maxMaxLevel = compileIndexAux(i, i == endIndex-1, maxMaxLevel, false);
            }
        }
        maxLevel = maxMaxLevel + 1;
    }

    private int compileIndexAux(int index, boolean last, int maxMaxLevel, boolean reverse) throws JSONPathException {
        int startLevel = ir.getCurrentLevel();
        if (reverse)
            ir.reverseIndex(index);
        else
            ir.index(index);
        ir.down();
        scanner.mark();
        int currentMaxLevel = maxLevel;
        if (scanner.hasNext()) {
            compileNextExpression();
        }
        maxMaxLevel = Math.max(maxLevel, maxMaxLevel);
        maxLevel = currentMaxLevel;
        if (last) {
            return maxMaxLevel;
        } else {
            ir.storeResult();
        }
        scanner.reset();
        int endLevel = ir.getCurrentLevel();
        for (int j = 0; j < endLevel - startLevel; j++) {
            ir.up();
        }
        return maxMaxLevel;
    }

    private void compileFilterExpression() throws JSONPathException {
        scanner.expectChar('?');
        scanner.expectChar('(');
        scanner.expectChar('@');
        while (scanner.skipIfChar(' ')) {
            // Skip whitespace
        }
        switch (scanner.peek()) {
            case '=':
                scanner.expectChar('=');
                scanner.expectChar('=');
                while (scanner.skipIfChar(' ')) {
                    // Skip whitespace
                }
                String equalTo = readQuotedString();
                ir.expressionStringEquals(equalTo);
                break;
            default:
                throw scanner.unsupportedNext("Unsupported character for expression");
        }
        scanner.expectChar(')');
    }

    private void createPropertyIR(String propertyName) {
        ir.property(propertyName);
        ir.down();
        maxLevel++;
    }

    private void createIndexIR(int index) {
        ir.index(index);
        ir.down();
        maxLevel++;
    }

    private void createReverseIndexIR(int index) {
        ir.reverseIndex(index);
        ir.down();
        maxLevel++;
    }

    private String readProperty() throws JSONPathException {
        int startPosition = scanner.position();
        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (c == ' ') {
                throw scanner.errorNext("Unexpected space");
            } else if (c == '.' || c == '[') {
                break;
            }
            scanner.next();
        }
        int endPosition = scanner.position();
        return scanner.substring(startPosition, endPosition);
    }

    private String readQuotedString() throws JSONPathException {
        char quoteCharacter = scanner.next();
        if (quoteCharacter != '\'' && quoteCharacter != '"') {
            throw scanner.error("Invalid quoted string");
        }
        int startPosition = scanner.position();
        boolean escaped = false;
        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == quoteCharacter) {
                break;
            }
            scanner.next();
        }
        int endPosition = scanner.position();
        scanner.expectChar(quoteCharacter);
        return scanner.substring(startPosition, endPosition);
    }

    private int readInteger(Predicate<Character> isEndCharacter) throws JSONPathException {
        int startPosition = scanner.position();
        while (scanner.hasNext()) {
            char c = scanner.peek();
            if (c >= '0' && c <= '9') {
                scanner.next();
                continue;
            } else if (isEndCharacter.test(c)) {
                break;
            }
            throw scanner.error("Invalid integer");
        }
        int endPosition = scanner.position();
        String str = scanner.substring(startPosition, endPosition);
        return Integer.parseInt(str);
    }
}
