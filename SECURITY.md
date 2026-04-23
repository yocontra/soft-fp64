# Security

soft-fp64 is a pure-compute math library. It performs no I/O, no heap
allocation, no threading, and has no global state. There is no traditional
attack surface: if a caller feeds attacker-controlled `double` values into
`sf64_*`, the results are whatever IEEE-754 dictates — there are no
memory-safety, injection, or privilege-boundary issues to worry about. The
library is fuzzed nightly via libFuzzer per-op targets under ASan + UBSan,
and exhaustive `f32 ↔ f64` round-trip is checked on every CI run.

If you believe you have found a real vulnerability — for example undefined
behavior on specific inputs, a buffer overflow in a helper, or a signed
overflow trip under UBSan — please report it by opening a GitHub issue or
emailing **yo@contra.io**. Bug reports from fuzzer corpora are
welcome.

## Supported Versions

Pre-1.0, unreleased. `main` is the only supported branch.
