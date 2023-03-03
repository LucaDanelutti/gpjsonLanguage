package it.necst.gpjson.jsonpath;

public class JSONPathQuery {
    private final IRByteOutputBuffer ir;
    private final int maxDepth;
    private final int numResults;

    public JSONPathQuery(IRByteOutputBuffer ir, int maxDepth, int numResults) {
        this.ir = ir;
        this.maxDepth = maxDepth;
        this.numResults = numResults;
    }

    public IRByteOutputBuffer getIr() {
        return ir;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public int getNumResults() {
        return numResults;
    }
}
