package it.necst.gpjson.utils;

import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

public class UnsafeHelper {
    private static final Unsafe unsafe;
    private static final long bufferAddressOffset;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            unsafe = (Unsafe) f.get(null);
            bufferAddressOffset = unsafe.objectFieldOffset(Buffer.class.getDeclaredField("address"));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // this needs to be a RuntimeException since it is raised during static initialization
            throw new RuntimeException(e);
        }
    }

    abstract public static class MemoryObject implements java.io.Closeable {
        private final long address;

        MemoryObject(long address) {
            this.address = address;
        }

        public final long getAddress() {
            return address;
        }

        @Override
        public void close() {
            unsafe.freeMemory(address);
        }
    }

    public static final class ByteArray extends MemoryObject {
        private final int numElements;

        ByteArray(int numElements) {
            super(unsafe.allocateMemory(numElements));
            this.numElements = numElements;
        }

        ByteArray(long address, int numElements) {
            super(address);
            this.numElements = numElements;
        }

        public void setValueAt(int index, byte value) {
            if ((index < 0) || (index >= numElements)) {
                throw new IllegalArgumentException(index + " is out of range");
            }
            unsafe.putByte(getAddress() + index, value);
        }
    }

    public static final class LongArray extends MemoryObject {
        private final long numElements;

        LongArray(long numElements) {
            super(unsafe.allocateMemory(8 * numElements));
            this.numElements = numElements;
        }

        LongArray(long address, int numElements) {
            super(address);
            this.numElements = numElements;
        }

        public long getValueAt(long index) {
            if ((index < 0) || (index >= numElements)) {
                throw new IllegalArgumentException(index + " is out of range");
            }
            return unsafe.getLong(getAddress() + 8*index);
        }

        public void setValueAt(long index, long value) {
            if ((index < 0) || (index >= numElements)) {
                throw new IllegalArgumentException(index + " is out of range");
            }
            unsafe.putLong(getAddress() + 8*index, value);
        }
    }

    public static final class IntArray extends MemoryObject {
        private final long numElements;

        IntArray(long numElements) {
            super(unsafe.allocateMemory(4 * numElements));
            this.numElements = numElements;
        }

        IntArray(long address, int numElements) {
            super(address);
            this.numElements = numElements;
        }

        public int getValueAt(long index) {
            if ((index < 0) || (index >= numElements)) {
                throw new IllegalArgumentException(index + " is out of range");
            }
            return unsafe.getInt(getAddress() + 4*index);
        }

        public void setValueAt(long index, int value) {
            if ((index < 0) || (index >= numElements)) {
                throw new IllegalArgumentException(index + " is out of range");
            }
            unsafe.putInt(getAddress() + 4*index, value);
        }
    }

    public static IntArray createIntArray(long numElements) {
        return new IntArray(numElements);
    }

    public static LongArray createLongArray(long numElements) {
        return new LongArray(numElements);
    }

    public static ByteArray createByteArray(ByteBuffer byteBuffer) {
        long address = unsafe.getLong(byteBuffer, bufferAddressOffset);
        return new ByteArray(address, byteBuffer.capacity());
    }

    public static LongArray createLongArray(LongBuffer longBuffer) {
        long address = unsafe.getLong(longBuffer, bufferAddressOffset);
        return new LongArray(address, longBuffer.capacity());
    }
}
