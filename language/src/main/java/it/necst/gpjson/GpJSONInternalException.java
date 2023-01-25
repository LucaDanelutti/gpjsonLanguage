package it.necst.gpjson;

import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.nodes.Node;

public class GpJSONInternalException extends AbstractTruffleException {
    public GpJSONInternalException(String message) {
        this(message, null);
    }

    public GpJSONInternalException(String message, Node node) {
        super(message, node);
    }

    public GpJSONInternalException(InteropException e) {
        this(e.getMessage());
    }
}
