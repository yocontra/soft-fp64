// Thread-safety / re-entrancy test for soft-fp64.
//
// README.md states: "Every symbol is a pure extern \"C\" function with
// no global state." This test gates that claim by running a representative
// slice of the sf64_* surface (arithmetic, sqrt, fma, and one op from
// each transcendental family — trig, exp/log, pow) once on the calling
// thread (reference), then running it concurrently on N_THREADS worker
// threads and asserting every worker's per-input output is bit-identical
// to the reference. Any op family sharing mutable state would trip here.
//
// A failure means a soft_fp64 op reads or writes shared mutable state
// (static cache, lazy-init, errno-alike). Detected here, not in prod.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int N_THREADS = 8;
constexpr int N_ITERS_PER_THREAD = 4;

inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

// A deterministic, numerically varied input corpus that touches the
// major sf64_* code paths (including subnormals, infinities, NaN, and
// ordinary values). Reused by both the reference pass and the worker
// threads.
std::vector<double> build_inputs() {
    std::vector<double> xs;
    for (double v : host_oracle::edge_cases_f64()) {
        xs.push_back(v);
    }
    // Add finite magnitudes across the interesting ranges for the
    // transcendentals we exercise below.
    for (int k = -6; k <= 6; ++k) {
        const double m = static_cast<double>(k) * 0.25 + 1e-3;
        xs.push_back(m);
        xs.push_back(-m);
    }
    return xs;
}

struct Outputs {
    std::vector<uint64_t> add, sub, mul, div, fma, sqrt_, sin_, cos_, tan_, exp_, log_, pow_;
};

// Runs the full op surface over `xs` and captures each result as raw
// bits. Bit-exact capture is the only way to detect a NaN-payload or
// signed-zero drift across threads.
void run_once(const std::vector<double>& xs, Outputs& out) {
    const std::size_t n = xs.size();
    out.add.resize(n);
    out.sub.resize(n);
    out.mul.resize(n);
    out.div.resize(n);
    out.fma.resize(n);
    out.sqrt_.resize(n);
    out.sin_.resize(n);
    out.cos_.resize(n);
    out.tan_.resize(n);
    out.exp_.resize(n);
    out.log_.resize(n);
    out.pow_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = xs[i];
        const double y = xs[(i + 7) % n];
        const double z = xs[(i + 13) % n];
        out.add[i] = bits(sf64_add(x, y));
        out.sub[i] = bits(sf64_sub(x, y));
        out.mul[i] = bits(sf64_mul(x, y));
        out.div[i] = bits(sf64_div(x, y));
        out.fma[i] = bits(sf64_fma(x, y, z));
        out.sqrt_[i] = bits(sf64_sqrt(x));
        out.sin_[i] = bits(sf64_sin(x));
        out.cos_[i] = bits(sf64_cos(x));
        out.tan_[i] = bits(sf64_tan(x));
        out.exp_[i] = bits(sf64_exp(x));
        out.log_[i] = bits(sf64_log(x));
        out.pow_[i] = bits(sf64_pow(x, y));
    }
}

bool same(const Outputs& a, const Outputs& b) {
    return a.add == b.add && a.sub == b.sub && a.mul == b.mul && a.div == b.div && a.fma == b.fma &&
           a.sqrt_ == b.sqrt_ && a.sin_ == b.sin_ && a.cos_ == b.cos_ && a.tan_ == b.tan_ &&
           a.exp_ == b.exp_ && a.log_ == b.log_ && a.pow_ == b.pow_;
}

} // namespace

int main() {
    const std::vector<double> xs = build_inputs();

    Outputs ref;
    run_once(xs, ref);

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    workers.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        workers.emplace_back([&xs, &ref, &mismatches] {
            for (int k = 0; k < N_ITERS_PER_THREAD; ++k) {
                Outputs got;
                run_once(xs, got);
                if (!same(got, ref)) {
                    mismatches.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }

    const int m = mismatches.load();
    if (m != 0) {
        std::fprintf(stderr,
                     "reentrancy: %d thread iterations diverged from the "
                     "single-threaded reference — sf64_* is NOT pure\n",
                     m);
        return 1;
    }
    std::printf("reentrancy: %d threads × %d iters × %zu inputs bit-identical\n", N_THREADS,
                N_ITERS_PER_THREAD, xs.size());
    return 0;
}
