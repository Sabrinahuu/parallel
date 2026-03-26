#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include<string>


#if defined(USE_ITT)
#  if defined(__has_include)
#    if __has_include(<ittnotify.h>)
#      include <ittnotify.h>
#      define HAS_ITT_HEADER 1
#    else
#      define HAS_ITT_HEADER 0
#    endif
#  else
#    include <ittnotify.h>
#    define HAS_ITT_HEADER 1
#  endif
#else
#  define HAS_ITT_HEADER 0
#endif

using namespace std;
using namespace std::chrono;

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

#if HAS_ITT_HEADER
static __itt_domain* g_domain = __itt_domain_create("sum.superscalar");

struct VtuneTask {
    __itt_string_handle* handle;
    explicit VtuneTask(const char* name) : handle(__itt_string_handle_create(name)) {
        __itt_task_begin(g_domain, __itt_null, __itt_null, handle);
    }
    ~VtuneTask() { __itt_task_end(g_domain); }
};

#define VTUNE_TASK(name) VtuneTask vtune_task_obj(name)
#define VTUNE_RESUME() __itt_resume()
#define VTUNE_PAUSE()  __itt_pause()
#else
struct VtuneTask {
    explicit VtuneTask(const char*) {}
};
#define VTUNE_TASK(name) VtuneTask vtune_task_obj(name)
#define VTUNE_RESUME() do {} while (0)
#define VTUNE_PAUSE()  do {} while (0)
#endif

NOINLINE long long naive_sum(const int* a, size_t n) {
    long long sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

NOINLINE long long way2_sum(const int* a, size_t n) {
    long long s0 = 0, s1 = 0;
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        s0 += a[i];
        s1 += a[i + 1];
    }
    if (i < n) s0 += a[i];
    return s0 + s1;
}

NOINLINE long long way4_sum(const int* a, size_t n) {
    long long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    for (; i < n; ++i) s0 += a[i];
    return s0 + s1 + s2 + s3;
}

NOINLINE long long way8_sum(const int* a, size_t n) {
    long long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    long long s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        s0 += a[i];     s1 += a[i + 1];
        s2 += a[i + 2]; s3 += a[i + 3];
        s4 += a[i + 4]; s5 += a[i + 5];
        s6 += a[i + 6]; s7 += a[i + 7];
    }
    for (; i < n; ++i) s0 += a[i];
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

NOINLINE long long way16_sum(const int* a, size_t n) {
    long long s[16] = {0};
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        for (int k = 0; k < 16; ++k) {
            s[k] += a[i + k];
        }
    }
    for (; i < n; ++i) s[0] += a[i];

    long long total = 0;
    for (int k = 0; k < 16; ++k) total += s[k];
    return total;
}

using SumFunc = long long (*)(const int*, size_t);

NOINLINE long long run_kernel(const char* name, SumFunc f, const int* a, size_t n, int repeat) {
    VTUNE_TASK(name);
    volatile long long sink = 0;
    for (int i = 0; i < repeat; ++i) {
        sink = f(a, n);
    }
    return sink;
}

static size_t parse_size(const char* s, size_t default_value) {
    if (s == nullptr) return default_value;
    try {
        return static_cast<size_t>(stoull(s));
    } catch (...) {
        return default_value;
    }
}

static int parse_int(const char* s, int default_value) {
    if (s == nullptr) return default_value;
    char* end = nullptr;
    long value = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || value <= 0) return default_value;
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    const size_t n = parse_size(argc > 1 ? argv[1] : nullptr, 1ull << 24);
    const int repeat = parse_int(argc > 2 ? argv[2] : nullptr, 200);

    vector<int> a(n, 1);
    const long long expected = static_cast<long long>(n);

    volatile long long warm = 0;
    for (int i = 0; i < 5; ++i) {
        warm = naive_sum(a.data(), n);
        warm = way2_sum(a.data(), n);
        warm = way4_sum(a.data(), n);
        warm = way8_sum(a.data(), n);
        warm = way16_sum(a.data(), n);
    }
    (void)warm;

    VTUNE_RESUME();
    const auto t0 = high_resolution_clock::now();

    const long long r0 = run_kernel("naive", naive_sum, a.data(), n, repeat);
    const long long r1 = run_kernel("way2", way2_sum, a.data(), n, repeat);
    const long long r2 = run_kernel("way4", way4_sum, a.data(), n, repeat);
    const long long r3 = run_kernel("way8", way8_sum, a.data(), n, repeat);
    const long long r4 = run_kernel("way16", way16_sum, a.data(), n, repeat);

    const auto t1 = high_resolution_clock::now();
    VTUNE_PAUSE();

    if (r0 != expected || r1 != expected || r2 != expected || r3 != expected || r4 != expected) {
        cerr << "wrong result: expected=" << expected
             << ", naive=" << r0
             << ", way2=" << r1
             << ", way4=" << r2
             << ", way8=" << r3
             << ", way16=" << r4 << '\n';
        return 1;
    }

    const double elapsed_ms = duration<double, milli>(t1 - t0).count();
    cout << "n=" << n << ", repeat=" << repeat << ", elapsed_ms=" << elapsed_ms << '\n';
#if HAS_ITT_HEADER
    cout << "ITT task markers enabled for VTune." << '\n';
#else
    cout << "ITT task markers disabled; code still works in VTune, but tasks will not be labeled unless compiled with -DUSE_ITT and linked with ITT." << '\n';
#endif
    cout << "All results are correct." << '\n';
    return 0;
}