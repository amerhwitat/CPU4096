# CPU4096

Simulator and research implementation of configurable wide-register CPUs, including the 4096-bit lineage and C8192/R8192 workload mapping.

## Language-separated implementation matrix

- `cpp/` — native C++ wide-register implementation and performance/conformance layer.
- `java/` — Java semantic/interoperability layer.
- `node/` — Node.js ESM integration and reproducibility layer.
- existing Python artifacts — research/reference workloads.

## Build and run

Native C++:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

Windows/MSVC, if the native build layer exists:

```bat
build-tools\build.bat --only native
```

Java:

```bash
cd java && mvn test
```

Node.js:

```bash
cd node && npm ci && npm test
```

Python reference workloads:

```bash
python -m pytest -q
```

## Register model

The simulator uses fixed-width registers represented as arrays of 64-bit host words. Widths are required to be multiples of 64 and may be extended to 8192-bit and beyond through templates/semantic models. Arithmetic is deterministic and wraps at the selected width.

## Chimera II integration

C8192/R8192 are architectural research targets. The repository supplies deterministic workloads to Chimera II OS and CPU4096Simulator; it does not claim physical 4096/8192-bit silicon.

## Security boundary

Cryptographic benchmarking uses public or synthetic material. No address-targeted private-key enumeration, seed guessing, credential harvesting or unauthorized wallet access is implemented.

See `docs/CRYPTO_AI_WORKLOADS.md` and the language READMEs for detailed contracts.
