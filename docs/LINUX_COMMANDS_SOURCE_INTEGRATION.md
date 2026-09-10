# Linux Commands Source Integration

Chimera II integration reference: GNU Coreutils, util-linux, iproute2/net-tools, sudo, POSIX shells, and Toybox. Preserve upstream licensing and SPDX metadata; use adapters for Chimera-specific interfaces.

Performance: bounded worker pools, batching, asynchronous I/O and deterministic single-thread fallbacks. CPU-heavy independent work may use parallel execution; stateful command semantics remain ordered.

Canonical standalone and Aurora Web UI integration is maintained in `amerhwitat/ChimeraIIOS`.
