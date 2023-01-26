package it.necst.gpjson.engine;

import it.necst.gpjson.engine.ExecutionContext;
import org.graalvm.polyglot.Value;

import java.util.Map;

public class ExecutionContextUncombined extends ExecutionContext {
    public ExecutionContextUncombined(Value cu, Map<String,Value> kernels, String fileName) {
        super(cu, kernels, fileName);
    }

    protected void createNewlineStringIndex() {
        stringIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        Value stringCarryIndexMemory = cu.invokeMember("DeviceArray", "char", gridSize * blockSize);
        Value newlineCountIndexMemory = cu.invokeMember("DeviceArray", "int", gridSize * blockSize);
        kernels.get("count_newlines").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineCountIndexMemory);
        kernels.get("create_escape_carry_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory);
        int sum = 1;
        for (int i=0; i<newlineCountIndexMemory.getArraySize(); i++) {
            int val = newlineCountIndexMemory.getArrayElement(i).asInt();
            newlineCountIndexMemory.setArrayElement(i, sum);
            sum += val;
        }
        newlineIndexMemory = cu.invokeMember("DeviceArray", "long", sum);
        Value escapeIndexMemory = cu.invokeMember("DeviceArray", "long", levelSize);
        kernels.get("create_escape_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), stringCarryIndexMemory, escapeIndexMemory, levelSize);
        kernels.get("create_newline_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), newlineCountIndexMemory, newlineIndexMemory);
        kernels.get("create_quote_index").execute(gridSize, blockSize).execute(fileMemory, fileMemory.getArraySize(), escapeIndexMemory, stringIndexMemory, stringCarryIndexMemory, levelSize);
        byte prev = 0;
        for (int i=0; i<stringCarryIndexMemory.getArraySize(); i++) {
            byte value = (byte) (stringCarryIndexMemory.getArrayElement(i).asByte() ^ prev);
            stringCarryIndexMemory.setArrayElement(i, value);
            prev = value;
        }
        kernels.get("create_string_index").execute(gridSize, blockSize).execute(levelSize, stringIndexMemory, stringCarryIndexMemory);
    }
}
