# CPU4096

Simulator and research implementation of configurable wide-register CPUs, including the 4096-bit lineage and C8192/R8192 workload mapping.

## Cross-language implementation matrix

- `cpp/` — native C++ wide-register implementation and performance/conformance layer.
- `java/` — Java semantic/interoperability layer.
- `node/` — Node.js ESM integration and reproducibility layer.
- existing Python artifacts — research/reference workloads.
- `chimera/` — common 128D state and P2P interoperability contract.

All tracks consume deterministic crypto/AI vectors and reproducibility metadata. The shared solution covers fixed-width arithmetic, bitwise operations, shifts, multiplication, register snapshots, concurrency experiments, SHA/Keccak, CNN/GRU/RNN and offline-RL workload metadata.

## Register model

The simulator uses fixed-width registers represented as arrays of 64-bit host words. Widths are required to be multiples of 64 and may be extended to 8192-bit and beyond through templates/semantic models. Arithmetic is deterministic and wraps at the selected width.

## Chimera 128D + P2P integration

CPU/register snapshots can be associated with the common Chimera 128D application-state envelope: geometry, time, observer/perspective, events, object properties, interactions and extensible vector state. `chimera/p2p_protocol.json` defines authenticated peer capability exchange and content-addressed synchronization for distributed simulator nodes. No arbitrary network scanning is implied.

## Chimera II integration

C8192/R8192 are architectural research targets. The repository supplies deterministic workloads to Chimera II OS and CPU4096Simulator; it does not claim physical 4096/8192-bit silicon.

## Security boundary

Cryptographic benchmarking uses public or synthetic material. No address-targeted private-key enumeration, seed guessing, credential harvesting or unauthorized wallet access is implemented.

## Build

C++:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

Java:

```bash
cd java && mvn test
```

Node.js:

```bash
cd node && npm test
```

See `docs/CRYPTO_AI_WORKLOADS.md` and the language READMEs for detailed contracts.

## License

Original project code is released under the GNU General Public License v3 or later. Third-party components retain their applicable licenses.
