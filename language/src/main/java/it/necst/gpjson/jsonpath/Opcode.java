package it.necst.gpjson.jsonpath;

public enum Opcode {
    END,
    STORE_RESULT,
    MOVE_UP,
    MOVE_DOWN,
    MOVE_TO_KEY,
    MOVE_TO_INDEX,
    MOVE_TO_INDEX_REVERSE,
    MARK_POSITION,
    RESET_POSITION,
    EXPRESSION_STRING_EQUALS
}
