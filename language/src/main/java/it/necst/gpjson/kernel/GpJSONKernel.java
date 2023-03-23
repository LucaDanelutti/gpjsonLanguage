package it.necst.gpjson.kernel;

public enum GpJSONKernel {
    // Uncombined
    NEWLINE_COUNT_INDEX("newline-count-index", "f(file: in pointer char, fileSize: sint32, newlineCountIndex: out pointer sint32)", "it/necst/gpjson/kernels/uncombined/newline-count-index.cu"),
    NEWLINE_INDEX("newline-index", "f(file: in pointer char, fileSize: sint32, newlineCountIndex: in pointer sint32, newlineIndex: out pointer sint64)", "it/necst/gpjson/kernels/uncombined/newline-index.cu"),
    ESCAPE_CARRY_INDEX("escape-carry-index", "f(file: in pointer char, fileSize: sint32, escapeCarryIndex: out pointer char)", "it/necst/gpjson/kernels/uncombined/escape-carry-index.cu"),
    ESCAPE_INDEX("escape-index", "f(file: in pointer char, fileSize: sint32, escapeCarryIndex: in pointer char, escapeIndex: out pointer sint64)", "it/necst/gpjson/kernels/uncombined/escape-index.cu"),
    // Combined
    COMBINED_ESCAPE_CARRY_NEWLINE_COUNT_INDEX("combined-escape-carry-newline-count-index", "f(file: in pointer char, fileSize: sint32, escapeCarryIndex: out pointer char, newlineCountIndex: out pointer sint32)", "it/necst/gpjson/kernels/combined-escape-carry-newline-count-index.cu"),
    INT_SUM_PRE_SCAN("int-sum-pre-scan", "f(intArr: inout pointer sint32, n: sint32)", "it/necst/gpjson/kernels/int-sum-pre-scan.cu"),
    INT_SUM_POST_SCAN("int-sum-post-scan", "f(intArr: in pointer sint32, n: sint32, stride: sint32, startingValue: sint32, base: out pointer sint32)", "it/necst/gpjson/kernels/int-sum-post-scan.cu"),
    INT_SUM_REBASE("int-sum-rebase", "f(intArr: in pointer sint32, n: sint32, base: in pointer sint32, offset: sint32, intNewArr: out pointer sint32)", "it/necst/gpjson/kernels/int-sum-rebase.cu"),
    COMBINED_ESCAPE_NEWLINE_INDEX("combined-escape-newline-index", "f(file: in pointer char, fileSize: sint32, escapeCarryIndex: in pointer char, newlineCountIndex: in pointer sint32, escapeIndex: out pointer sint64, newlineIndex: out pointer sint64)", "it/necst/gpjson/kernels/combined-escape-newline-index.cu"),
    QUOTE_INDEX("quote-index", "f(file: in pointer char, fileSize: sint32, escapeIndex: in pointer sint64, quoteIndex: out pointer sint64, quoteCarryIndex: out pointer char)", "it/necst/gpjson/kernels/quote-index.cu"),
    XOR_PRE_SCAN("xor-pre-scan", "f(charArr: inout pointer char, n: sint32)", "it/necst/gpjson/kernels/xor-pre-scan.cu"),
    XOR_POST_SCAN("xor-post-scan", "f(charArr: in pointer char, n: sint32, stride: sint32, base: out pointer char)", "it/necst/gpjson/kernels/xor-post-scan.cu"),
    XOR_REBASE("xor-rebase", "f(charArr: inout pointer char, n: sint32, base: in pointer char)", "it/necst/gpjson/kernels/xor-rebase.cu"),
    STRING_INDEX("string-index", "f(quoteIndex: out pointer sint64, quoteIndexSize: sint32, quoteCarryIndex: in pointer char)", "it/necst/gpjson/kernels/string-index.cu"),
    LEVELED_BITMAPS_CARRY_INDEX("leveled-bitmaps-carry-index", "f(file: in pointer char, fileSize: sint32, stringIndex: in pointer sint64, leveledBitmapsAuxIndex: out pointer sint8)", "it/necst/gpjson/kernels/leveled-bitmaps-carry-index.cu"),
    CHAR_SUM_PRE_SCAN("char-sum-pre-scan", "f(charArr: inout pointer char, n: sint32)", "it/necst/gpjson/kernels/char-sum-pre-scan.cu"),
    CHAR_SUM_POST_SCAN("char-sum-post-scan", "f(charArr: in pointer char, n: sint32, stride: sint32, startingValue: char, base: out pointer char)", "it/necst/gpjson/kernels/char-sum-post-scan.cu"),
    CHAR_SUM_REBASE("char-sum-rebase", "f(charArr: in pointer char, n: sint32, base: in pointer char, offset: sint32, charNewArr: out pointer char)", "it/necst/gpjson/kernels/char-sum-rebase.cu"),
    LEVELED_BITMAPS("leveled-bitmaps-index", "f(file: in pointer char, fileSize: sint32, stringIndex: in pointer sint64, leveledBitmapsAuxIndex: in pointer sint8, leveledBitmapsIndex: out pointer sint64, levelSize: sint32, numLevels: sint32)", "it/necst/gpjson/kernels/leveled-bitmaps-index.cu"),
    QUERY("query", "f(file: in pointer char, fileSize: sint32, newlineIndex: in pointer sint64, newlineIndexSize: sint32, stringIndex: in pointer sint64, leveledBitmapsIndex: in pointer sint64, levelSize: sint32, query: in pointer char, numResults: sint32, result: out pointer sint64)", "it/necst/gpjson/kernels/query.cu");

    private final String name;
    private final String parameterSignature;
    private final String filename;

    GpJSONKernel(String name, String parameterSignature, String filename) {
        this.name = name;
        this.parameterSignature = parameterSignature;
        this.filename = filename;
    }

    public String getName() {
        return name;
    }

    public String getParameterSignature() {
        return parameterSignature;
    }

    public String getFilename() {
        return filename;
    }
}

