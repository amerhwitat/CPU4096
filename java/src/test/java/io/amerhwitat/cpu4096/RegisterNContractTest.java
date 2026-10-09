package io.amerhwitat.cpu4096;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class RegisterNContractTest {
    private static final Path FIXTURES = Path.of("..", "contracts", "fixtures", "cpu4096-register-v1.tsv");

    @Test
    void sharedRegisterVectorsPass() throws IOException {
        int count = 0;
        for (String line : Files.readAllLines(FIXTURES, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] f = line.split("\\t", -1);
            assertEquals(6, f.length, "malformed fixture: " + line);
            int width = Integer.parseInt(f[1]);
            RegisterN a = new RegisterN(width, new BigInteger(f[2]));
            RegisterN b = new RegisterN(width, new BigInteger(f[3]));
            int shift = Integer.parseInt(f[4]);
            switch (f[0]) {
                case "add" -> assertEquals(new BigInteger(f[5]), a.add(b).unsignedValue());
                case "wrap-add" -> assertEquals(new BigInteger(f[5]), a.add(b).unsignedValue());
                case "subtract-wrap" -> assertEquals(new BigInteger(f[5]), a.subtract(b).unsignedValue());
                case "shift-left" -> assertEquals(new BigInteger(f[5]), a.shiftLeft(shift).unsignedValue());
                case "shift-right" -> assertEquals(new BigInteger(f[5]), a.shiftRight(shift).unsignedValue());
                case "snapshot" -> assertEquals(width + ":" + String.join(",", a.snapshotWords()), f[5]);
                default -> throw new AssertionError("unknown fixture: " + f[0]);
            }
            count++;
        }
        assertEquals(6, count);
    }

    @Test
    void rejectsInvalidWidthsAndMixedWidthArithmetic() {
        assertThrows(IllegalArgumentException.class, () -> new RegisterN(65, BigInteger.ZERO));
        assertThrows(IllegalArgumentException.class,
            () -> new RegisterN(64, BigInteger.ONE).add(new RegisterN(128, BigInteger.ONE)));
    }

    @Test
    void snapshotsAreLittleEndianWordsAndStable() {
        RegisterN r = new RegisterN(128, new BigInteger("36893488147419103231"));
        assertEquals(List.of("18446744073709551615", "1"), r.snapshotWords());
        assertEquals(r.snapshotWords(), r.snapshotWords());
    }
}
