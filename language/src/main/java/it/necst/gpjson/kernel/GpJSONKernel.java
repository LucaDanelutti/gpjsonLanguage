package it.necst.gpjson.kernel;

public enum GpJSONKernel {
    // Uncombined
    COUNT_NEWLINES("count_newlines", "count_newlines(file: in pointer char, n: sint64, result: out pointer sint32)", "it/necst/gpjson/kernels/uncombined/count_newlines.cu"),
    CREATE_NEWLINE_INDEX("create_newline_index", "create_newline_index(file: in pointer char, n: sint64, indices: in pointer sint32, result: out pointer sint64)", "it/necst/gpjson/kernels/uncombined/create_newline_index.cu"),
    CREATE_ESCAPE_CARRY_INDEX("create_escape_carry_index", "create_escape_carry_index(file: in pointer char, n: sint64, escape_carry_index: out pointer char)", "it/necst/gpjson/kernels/uncombined/create_escape_carry_index.cu"),
    CREATE_ESCAPE_INDEX("create_escape_index", "create_escape_index(file: in pointer char, n: sint64, escape_carry_index: in pointer char, escape_index: out pointer sint64, escape_index_size: sint64)", "it/necst/gpjson/kernels/uncombined/create_escape_index.cu"),
    // Combined
    CREATE_QUOTE_INDEX("create_quote_index", "create_quote_index(file: in pointer char, n: sint64, escape_index: in pointer sint64, quote_index: out pointer sint64, quote_carry_index: out pointer char, quote_index_size: sint64)", "it/necst/gpjson/kernels/create_quote_index.cu"),
    CREATE_STRING_INDEX("create_string_index", "create_string_index(n: sint64, quote_index: out pointer sint64, quote_counts: in pointer char)", "it/necst/gpjson/kernels/create_string_index.cu"),
    CREATE_LEVELED_BITMAPS_CARRY_INDEX("create_leveled_bitmaps_carry_index", "create_leveled_bitmaps_carry_index(file: in pointer char, n: sint64, string_index: in pointer sint64, level_carry_index: out pointer sint8)", "it/necst/gpjson/kernels/create_leveled_bitmaps_carry_index.cu"),
    CREATE_LEVELED_BITMAPS("create_leveled_bitmaps", "create_leveled_bitmaps(file: in pointer char, n: sint64, string_index: in pointer sint64, carry_index: in pointer sint8, leveled_bitmaps_index: out pointer sint64, leveled_bitmaps_index_size: sint64, level_size: sint64, num_levels: sint32)", "it/necst/gpjson/kernels/create_leveled_bitmaps.cu"),
    FIND_VALUE("find_value", "find_value(file: in pointer char, n: sint64, new_line_index: in pointer sint64, new_line_index_size: sint64, string_index: in pointer sint64, leveled_bitmaps_index: in pointer sint64, leveled_bitmaps_index_size: sint64, level_size: sint64, query: in pointer char, result_size: sint32, result: out pointer sint64)", "it/necst/gpjson/kernels/find_value.cu"),
    QUERY("executeQuery", "executeQuery(file: in pointer char, n: sint64, newlineIndex: in pointer sint64, newlineIndexSize: sint64, stringIndex: in pointer sint64, leveledBitmapsIndex: in pointer sint64, levelSize: sint64, query: in pointer char, numResults: sint32, result: out pointer sint64)", "it/necst/gpjson/kernels/query.cu"),
    CREATE_COMBINED_ESCAPE_CARRY_NEWLINE_COUNT_INDEX("create_combined_escape_carry_newline_count_index", "create_combined_escape_carry_newline_count_index(file: in pointer char, n: sint64, escape_carry_index: out pointer char, newline_count_index: out pointer sint32)", "it/necst/gpjson/kernels/create_combined_escape_carry_newline_count_index.cu"),
    CREATE_COMBINED_ESCAPE_NEWLINE_INDEX("create_combined_escape_newline_index", "create_combined_escape_newline_index(file: in pointer char, n: sint64, escape_carry_index: in pointer char, newline_count_index: in pointer sint32, escape_index: out pointer sint64, escape_index_size: sint64, newline_index: out pointer sint64)", "it/necst/gpjson/kernels/create_combined_escape_newline_index.cu"),
    INT_SUM1("int_sum1", "sum1(intArr: inout pointer sint32, n: sint32)", "it/necst/gpjson/kernels/int_sum1.cu"),
    INT_SUM2("int_sum2", "sum2(intArr: in pointer sint32, n: sint32, stride: sint32, startingValue: sint32, base: out pointer sint32)", "it/necst/gpjson/kernels/int_sum2.cu"),
    INT_SUM3("int_sum3", "sum3(intArr: in pointer sint32, n: sint32, base: in pointer sint32, offset: sint32, intNewArr: out pointer sint32)", "it/necst/gpjson/kernels/int_sum3.cu"),
    XOR1("xor1", "xor1(charArr: inout pointer char, n: sint32)", "it/necst/gpjson/kernels/xor1.cu"),
    XOR2("xor2", "xor2(charArr: in pointer char, n: sint32, stride: sint32, base: out pointer char)", "it/necst/gpjson/kernels/xor2.cu"),
    XOR3("xor3", "xor3(charArr: inout pointer char, n: sint32, base: in pointer char)", "it/necst/gpjson/kernels/xor3.cu"),
    CHAR_SUM1("char_sum1", "sum1(charArr: inout pointer char, n: sint32)", "it/necst/gpjson/kernels/char_sum1.cu"),
    CHAR_SUM2("char_sum2", "sum2(charArr: in pointer char, n: sint32, stride: sint32, startingValue: char, base: out pointer char)", "it/necst/gpjson/kernels/char_sum2.cu"),
    CHAR_SUM3("char_sum3", "sum3(charArr: in pointer char, n: sint32, base: in pointer char, offset: sint32, charNewArr: out pointer char)", "it/necst/gpjson/kernels/char_sum3.cu");

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

