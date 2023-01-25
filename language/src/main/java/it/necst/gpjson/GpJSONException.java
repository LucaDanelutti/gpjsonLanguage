package it.necst.gpjson;

import com.oracle.truffle.api.exception.AbstractTruffleException;
import com.oracle.truffle.api.interop.InteropException;
import com.oracle.truffle.api.nodes.Node;

public class GpJSONException extends AbstractTruffleException {

    public GpJSONException(String message) {
        this(message, null);
    }

    public GpJSONException(String message, Node node) {
        super(message, node);
    }

    public GpJSONException(InteropException e) {
        this(e.getMessage());
    }
}
