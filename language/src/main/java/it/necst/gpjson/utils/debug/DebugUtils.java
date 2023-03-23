package it.necst.gpjson.utils.debug;

import org.graalvm.polyglot.Value;

public class DebugUtils {
    public static void printDeviceArrayBinary(Value deviceArray, String name, int bits) {
        System.out.println(name + ":b:");
        for (int j=0; j<bits; j++) {
            System.out.print(((j+1) % 10) + "|");
        }
        System.out.println();
        for (int i=0; i<deviceArray.getArraySize(); i++) {
            long val = 1;
            for (int j=0; j<bits; j++) {
                int res = (deviceArray.getArrayElement(i).asLong() & val) != 0 ? 1 : 0;
                System.out.print(res + "|");
                val <<= 1;
            }
            System.out.println();
        }
    }

    public static void printDeviceArray(Value deviceArray, String name) {
        System.out.print(name + ":");
        for (int i=0; i<deviceArray.getArraySize(); i++) {
            System.out.print(deviceArray.getArrayElement(i).asInt() + "|");
        }
        System.out.println();
    }
}
