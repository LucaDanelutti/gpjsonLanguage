package it.necst.gpjson.utils;

import com.oracle.truffle.api.CompilerDirectives;
import it.necst.gpjson.GpJSONInternalException;
import it.necst.gpjson.engine.core.DataBuilder;

import java.nio.MappedByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class HashHelper {
    private static MessageDigest digest;

    private static void concatPartition(MappedByteBuffer buffer) {
        byte[] byteArray = new byte[1024];

        buffer.position(0);
        while (buffer.position() != buffer.capacity()) {
            buffer.get(byteArray, 0, Math.min(buffer.capacity()-buffer.position(), 1024));
            digest.update(byteArray, 0, Math.min(buffer.capacity()-buffer.position(), 1024));
        }
    }

    public static String computeHash(DataBuilder[] dataBuilder, int numPartitions) {
        try {
            digest = MessageDigest.getInstance("SHA-256");
            for (int i=0; i < numPartitions; i++) {
                concatPartition(dataBuilder[i].getFileBuffer());
            }
            byte[] bytes = digest.digest();
            StringBuilder sb = new StringBuilder();
            for (byte aByte : bytes) {
                sb.append(Integer.toString((aByte & 0xff) + 0x100, 16).substring(1));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GpJSONInternalException("No sha256 algorithm found");
        }
    }
}
