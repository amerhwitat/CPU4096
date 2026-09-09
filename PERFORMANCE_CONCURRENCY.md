# Performance & Concurrency Policy

This repository follows the Chimera II performance model: parallelize independent work, avoid shared mutable state, bound worker counts, and never assume more threads means more performance.

## Rules
- Prefer data-parallel work over shared-state worker threads.
- Use `std::thread`/parallel algorithms only for sufficiently large workloads; avoid thread creation inside hot inner loops.
- Cap workers at available hardware concurrency and workload size.
- Keep deterministic/single-thread modes available for debugging and conformance tests.
- Avoid oversubscription when BLAS/OpenMP/runtime libraries already create workers.
- Use immutable inputs, per-worker scratch state, and deterministic reduction where practical.
- Benchmark before/after changes; optimize measured hot paths rather than adding concurrency indiscriminately.

## 4096-bit runtime guidance
Large integer operations should use chunk-level parallelism only when operand size and operation cost justify synchronization overhead. Neural-network layers may parallelize independent neuron evaluation/update work, while training steps that mutate shared weights remain ordered unless a defined parallel-training algorithm is introduced.

## Safety
Concurrency changes must preserve ISA semantics, ABI behavior, numerical correctness, and reproducibility. Performance optimizations must not weaken cryptographic validation or introduce data races.
