# Security

soft-fp64 is a pure-compute math library. It performs no I/O or heap
allocation and has no global state. Attacker-controlled `double` inputs
to `sf64_*` produce the IEEE-754-defined result; there is no memory
safety, injection, or privilege boundary in scope. The library is
fuzzed nightly via libFuzzer per-op targets under ASan + UBSan, and
exhaustive `f32 ↔ f64` round-trip is checked on every CI run.

If you believe you have found a vulnerability (undefined behavior on
specific inputs, a buffer overflow in a helper, a signed overflow trip
under UBSan), report it by opening a GitHub issue or emailing
**yo@contra.io**. Bug reports from fuzzer corpora are welcome.

## Supported Versions

Pre-1.0, unreleased. `main` is the only supported branch.
