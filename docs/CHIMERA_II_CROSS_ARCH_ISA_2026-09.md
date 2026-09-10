# Chimera II C8192/R8192 Cross-Architecture ISA Expansion

This companion specification records the 2026 cross-architecture audit for the 4096/8192-bit Chimera register model. The canonical Chimera ABI remains unchanged.

## Covered architecture families

- x86-64 / IA-32: arithmetic, bit manipulation, string/memory, SIMD, atomics, fences, virtualization and system instructions.
- Arm A64/SVE/SME: acquire/release atomics, scalable vectors, predicates, gather/scatter, matrix operations, cache hints and system instructions.
- RISC-V: M/A/B/V/Zicsr/Zifencei/Zc/crypto and related ratified extensions.
- IBM POWER: user, virtual-environment and operating-environment instruction facilities, plus vector/dense-math/crypto directions.
- IBM z/Architecture: decimal, string, vector, address-space and system-control facilities.
- SPARC V9/VIS: 64-bit changes, branches, integer/FP, synthetic and vector/graphics instructions.
- HP PA-RISC: fixed-width load/store RISC, control, FP and multimedia/MAX facilities.
- Intel Itanium: EPIC bundles, predicate/branching, memory, system and IA-32 compatibility concepts.
- MIPS: fixed 32-bit load/store, HI/LO multiply/divide, branch and coprocessor concepts.

## Chimera extension classes

BITMANIP, ATOMICS, MEMORY_ORDER, VECTOR, MATRIX_TENSOR, CRYPTO, STRING_MEMORY, CACHE_MEMORY, SYSTEM_CONTROL, VIRTUALIZATION, DECIMAL, PREDICATE and NETWORK.

The implementation strategy is semantic normalization. Vendor instruction encodings are not copied into Chimera. Compiler backends and the assembler/disassembler may lower to the Chimera semantic operations, while C8192/R8192 keep their existing canonical encoding.

## Reference operations already implemented in the main OS

`CLZ`, `CTZ`, `POPCNT`, wide vector add, wide lane-wise multiply-add and CRC32C reference semantics are available through `chimera/isa_extension_ops.hpp` in ChimeraIIOS. The CPU4096 project remains the wide-register arithmetic and architecture research repository; ChimeraIIOS is the ABI/runtime integration point.

## Sources

- Intel SDM: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
- Arm A64: https://developer.arm.com/documentation/ddi0602/2026-06
- RISC-V: https://docs.riscv.org/reference/isa/
- OpenPOWER: https://openpowerfoundation.org/specifications/isa/
- IBM z/Architecture: https://www.ibm.com/docs/en/systems-hardware/zsystems/3932-A02?topic=library-2
- SPARC V9: https://docs.oracle.com/cd/E18752_01/html/816-1681/sparcv9-15322.html
- PA-RISC: https://parisc.docs.kernel.org/en/latest/technical_documentation.html
