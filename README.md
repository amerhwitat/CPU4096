# CPU4096

Simulator and research implementation of configurable wide-register CPUs, including the 4096-bit lineage and C8192/R8192 workload mapping.

## Source-code citation index

- [C++ implementation](cpp/)
- [Java implementation](java/)
- [Node.js implementation](node/)
- [Python/reference workloads](.)
- [.NET compatibility implementation](dotnet/)
- [Apple implementation](apple/)
- [Centralized Apple Objective-C + Flutter companion](https://github.com/amerhwitat/general/tree/master/Apple-Implementations/CPU4096)
- [Documentation](docs/)

## Apple Objective-C + Flutter

The centralized Apple companion is [`general/Apple-Implementations/CPU4096`](https://github.com/amerhwitat/general/tree/master/Apple-Implementations/CPU4096). It contains Objective-C native bridges, XcodeGen configuration and Flutter iOS/macOS sources for the simulator UI and native compute boundary.

## Cross-language implementation matrix

- `cpp/` — native C++ wide-register implementation and performance/conformance layer.
- `java/` — Java semantic/interoperability layer.
- `node/` — Node.js ESM integration and reproducibility layer.
- existing Python artifacts — research/reference workloads.
- `dotnet/` — managed compatibility layer where present.
- `apple/` — SwiftUI/Xcode iOS/iPadOS and macOS application boundary.

All tracks consume deterministic crypto/AI vectors and reproducibility metadata. The shared solution covers fixed-width arithmetic, bitwise operations, shifts, multiplication, register snapshots, concurrency experiments, SHA/Keccak, CNN/GRU/RNN and offline-RL workload metadata.

## Register model

The simulator uses fixed-width registers represented as arrays of 64-bit host words. Widths are required to be multiples of 64 and may be extended to 8192-bit and beyond through templates/semantic models. Arithmetic is deterministic and wraps at the selected width.

## Chimera 128D + P2P integration

CPU and workload state can be represented through the Chimera 128D semantic profile: geometry/state, time, observer/perspective, events, objects, properties and interaction rules, with an extensible perception/cognition layer. P2P synchronization is opt-in and authenticated, using deterministic snapshots/deltas, sequence numbers, payload hashes and capability exchange.

The P2P layer is for trusted simulator/application nodes and does not perform unsolicited network scanning, credential exchange, private-key discovery or arbitrary remote execution. See `docs/CHIMERA_128D_P2P_INTEGRATION.md`.

## Chimera II integration

C8192/R8192 are architectural research targets. The repository supplies deterministic workloads to Chimera II OS and CPU4096Simulator; it does not claim physical 4096/8192-bit silicon.

## Security boundary

Cryptographic benchmarking uses public or synthetic material. No address-targeted private-key enumeration, seed guessing, credential harvesting or unauthorized wallet access is implemented.

## Licensing

The repository is distributed under GNU GPL v3 or later. Existing third-party dependencies remain under their respective licenses.
