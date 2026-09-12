# Chimera 128D + authenticated P2P integration

CPU4096 exposes deterministic multidimensional CPU state for Chimera research.

The 128D envelope models register/CPU state together with geometry, time, observer/perspective, events, objects and interaction rules, with an extensible cognitive/perception overlay. It is a semantic model and does not imply physical 4096/8192-bit silicon.

The P2P layer is opt-in and authenticated. Nodes exchange identity, protocol version and capabilities, then synchronize deterministic snapshots/deltas using sequence numbers and payload hashes. Supported patterns are request/response, pub/sub and content-addressed synchronization.

Cross-language implementations must use the same logical envelope and conformance vectors. No unsolicited network scanning, credential exchange, private-key discovery or arbitrary remote execution is part of this integration.
