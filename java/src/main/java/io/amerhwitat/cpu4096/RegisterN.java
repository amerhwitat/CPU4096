package io.amerhwitat.cpu4096;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Unsigned fixed-width register matching the repository's RegisterN<Bits> contract.
 * Arithmetic is modulo 2^width; snapshot words are ordered least-significant first.
 */
public final class RegisterN {
    private final int widthBits;
    private final BigInteger value;

    public RegisterN(int widthBits, BigInteger value) {
        if (widthBits <= 0 || widthBits % 64 != 0) {
            throw new IllegalArgumentException("register width must be a positive multiple of 64");
        }
        this.widthBits = widthBits;
        this.value = Objects.requireNonNull(value, "value").mod(BigInteger.ONE.shiftLeft(widthBits));
    }

    public static RegisterN of(int widthBits, long value) {
        return new RegisterN(widthBits, BigInteger.valueOf(value).and(BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE)));
    }

    public int widthBits() { return widthBits; }
    public BigInteger unsignedValue() { return value; }
    public boolean isZero() { return value.signum() == 0; }
    public boolean msb() { return value.testBit(widthBits - 1); }

    private void requireSameWidth(RegisterN other) {
        Objects.requireNonNull(other, "other");
        if (widthBits != other.widthBits) throw new IllegalArgumentException("register widths differ");
    }

    public int compareUnsigned(RegisterN other) { requireSameWidth(other); return value.compareTo(other.value); }
    public RegisterN add(RegisterN other) { requireSameWidth(other); return new RegisterN(widthBits, value.add(other.value)); }
    public RegisterN subtract(RegisterN other) { requireSameWidth(other); return new RegisterN(widthBits, value.subtract(other.value)); }
    public RegisterN and(RegisterN other) { requireSameWidth(other); return new RegisterN(widthBits, value.and(other.value)); }
    public RegisterN or(RegisterN other) { requireSameWidth(other); return new RegisterN(widthBits, value.or(other.value)); }
    public RegisterN xor(RegisterN other) { requireSameWidth(other); return new RegisterN(widthBits, value.xor(other.value)); }
    public RegisterN not() { return new RegisterN(widthBits, value.xor(mask(widthBits))); }
    public RegisterN shiftLeft(int count) {
        if (count < 0) throw new IllegalArgumentException("negative shift count");
        return new RegisterN(widthBits, count >= widthBits ? BigInteger.ZERO : value.shiftLeft(count));
    }
    public RegisterN shiftRight(int count) {
        if (count < 0) throw new IllegalArgumentException("negative shift count");
        return new RegisterN(widthBits, count >= widthBits ? BigInteger.ZERO : value.shiftRight(count));
    }

    public List<String> snapshotWords() {
        BigInteger mask = mask(64);
        List<String> words = new ArrayList<>(widthBits / 64);
        for (int i = 0; i < widthBits / 64; i++) {
            words.add(value.shiftRight(i * 64).and(mask).toString());
        }
        return List.copyOf(words);
    }

    public String toHex() { return "0x" + value.toString(16).toUpperCase(); }
    private static BigInteger mask(int bits) { return BigInteger.ONE.shiftLeft(bits).subtract(BigInteger.ONE); }
}
