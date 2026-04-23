# soft-fp64 bench harness

Microbench + regression gate for the public `sf64_*` entry points. No
external dependencies.

## Build

From the repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOFT_FP64_BUILD_BENCH=ON
cmake --build build --target bench_soft_fp64
```

The binary lands at `build/bench/bench_soft_fp64`. `-DCMAKE_BUILD_TYPE=Release`
is mandatory — debug builds produce numbers that are not comparable to the
committed baseline.

## Run

```bash
# Pretty-print to stdout (default).
./build/bench/bench_soft_fp64

# Run only ops matching a substring.
./build/bench/bench_soft_fp64 --filter=log

# Longer per-op timing window (default 200ms).
./build/bench/bench_soft_fp64 --min-time-ms=1000

# Emit JSON for the regression gate. Schema is `soft-fp64.bench.v1`.
./build/bench/bench_soft_fp64 --json > /tmp/current.json
```

Flags:

| Flag                 | Default | Purpose                                             |
|----------------------|--------:|-----------------------------------------------------|
| `--min-time-ms=N`    |   `200` | Auto-scale iteration count to at least `N` ms/op.   |
| `--filter=substr`    |    none | Only run ops whose name contains `substr`.          |
| `--json`             |   false | Emit JSON to stdout instead of the human table.     |

## Regression gate (`compare.py`)

`compare.py` is a stdlib-only Python 3.10+ script. It compares a fresh JSON
run against the committed baseline and exits non-zero if any op is slower
by more than the threshold.

```bash
./build/bench/bench_soft_fp64 --json > /tmp/current.json
python3 bench/compare.py /tmp/current.json bench/baseline.json --threshold=0.05
echo $?   # 0 == no regressions, 1 == at least one op regressed
```

Output is a Markdown table sorted by `abs(delta)`, with regressions and
improvements called out separately. The gate:

- matches ops by `name`;
- skips ops whose baseline `ns_per_op` is below a 2 ns noise floor
  (sub-2ns timings swing too much run-to-run to be a reliable signal);
- treats ops missing from `current` as a note (not a failure) so you can
  run `--filter` subsets without tripping the gate;
- treats ops present in `current` but not in `baseline` as informational
  ("new op").

### CI usage

One-shot in CI:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOFT_FP64_BUILD_BENCH=ON
cmake --build build --target bench_soft_fp64 -j
./build/bench/bench_soft_fp64 --json --min-time-ms=500 > current.json
python3 bench/compare.py current.json bench/baseline.json --threshold=0.10
```

A 10% threshold is a reasonable default for shared CI runners (which
share cores with other jobs and thermally throttle). On a quiet local
machine 5% is tight enough to catch real regressions.

## Updating `baseline.json`

Only update the baseline when the change is **legitimate**:

- the host hardware changed (new CI runner image, new dev machine);
- an intentional, reviewed perf change landed (documented in the PR);
- the bench harness itself changed (new op added, methodology fixed).

Do **not** update the baseline to silence a regression on unchanged code.
If `compare.py` fails, the fix is in the library, not the baseline.

Procedure:

```bash
# 1. Close other heavy processes. Plug in (laptops throttle on battery).
# 2. Run 3x, keep the best (min ns/op per op is the most stable point
#    estimate — see "take min of k runs" in any perf-methodology doc).
for i in 1 2 3; do
  ./build/bench/bench_soft_fp64 --json --min-time-ms=1000 > /tmp/run.$i.json
done

# 3. Merge (stdlib, trivial):
python3 - <<'PY'
import json, glob
runs = [json.load(open(p)) for p in sorted(glob.glob('/tmp/run.*.json'))]
best = {}
for r in runs:
    for op in r['results']:
        prev = best.get(op['name'])
        if prev is None or op['ns_per_op'] < prev['ns_per_op']:
            best[op['name']] = op
out = {'schema': 'soft-fp64.bench.v1',
       'results': [best[k] for k in sorted(best)]}
print(json.dumps(out, indent=2))
PY
# 4. Review the diff vs the old baseline; commit with an explanation.
```

Record in the commit message which machine produced the numbers
(e.g. "Apple M3 Max, macOS 15.2, clang 18.1.8, Release").

## Comparative bench (`bench_compare`)

`bench_compare` measures `sf64_*` against two vendored reference soft-float
libraries, giving an external baseline for relative-perf claims:

- [Berkeley SoftFloat 3e](http://www.jhauser.us/arithmetic/SoftFloat.html) —
  the canonical bit-exact IEEE-754 reference. C API. Core IEEE ops only
  (no transcendentals).
- [ckormanyos/soft_double](https://github.com/ckormanyos/soft_double) — a
  header-only C++ double implementation with a full `<cmath>`-style
  surface (sin/cos/exp/log/pow/sqrt/…).

Neither is pulled via `FetchContent` — that would gate every clean build
on network access. Instead a one-shot helper clones them into
`bench/external/`:

```bash
./bench/fetch_external.sh    # clones softfloat/ and soft_double/ if missing
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOFT_FP64_BUILD_BENCH=ON
cmake --build build --target bench_compare -j
./build/bench/bench_compare --min-time-ms=500 --json > compare.json
```

The bench auto-detects which vendor libraries are present at configure
time (`bench/fetch_external.cmake`). If only one is vendored the bench
still builds, and the missing rows are simply absent from the output.

### Scope

| Op family      | sf64 | SoftFloat 3e | soft_double |
|----------------|:----:|:------------:|:-----------:|
| add/sub/mul/div/fma/sqrt | ✓ | ✓ | ✓ |
| sin/cos/tan/exp/log/pow/sinh/cosh/tanh | ✓ |   | ✓ |

SoftFloat 3e is deliberately IEEE-core-only upstream; `soft_double`
borrows its transcendentals from Boost.Math and is the only library with
a comparable transcendental surface. FMA on `soft_double` is emulated as
`a * b + c` (no native FMA entry point).

### CI

The `comparative-bench` job runs on macOS on `workflow_dispatch` and on
tag pushes (`v*`). It uploads `compare.json` as a workflow artifact and,
on tag pushes, attaches it to the GitHub Release. It is **not** a
regression gate — the numbers are informational and will drift with
upstream vendor updates.
