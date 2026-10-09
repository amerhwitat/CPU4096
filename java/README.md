# Java implementation

Java 21 GUI/benchmark integration layer for register-model visualization, deterministic vectors, experiment metadata, and interoperability with the Chimera stack.


## Java 21 register implementation

The source tree now contains `io.amerhwitat.cpu4096.RegisterN`, an unsigned fixed-width register model with modulo-​2^N arithmetic, bitwise operations, shifts, and stable least-significant-word-first snapshots. Shared cross-language acceptance vectors live at `../contracts/fixtures/cpu4096-register-v1.tsv`. Run `mvn --batch-mode --no-transfer-progress clean verify` from this directory. This is the register semantic layer, not a claim that the JavaFX visualization layer is implemented.
