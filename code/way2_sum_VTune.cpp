#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

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
static __itt_domain* g_domain = __itt_domain_create("sum.naive");

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

NOINLINE long long run_kernel(const char* name, const int* a, size_t n, int repeat) {
    VTUNE_TASK(name);
    volatile long long sink = 0;
    for (int i = 0; i < repeat; ++i) {
        sink = naive_sum(a, n);
    }
    return sink;
}

static size_t parse_size(const char* s, size_t default_value) {
    if (s == nullptr) return default_value;
    try {
        return static_cast<size_t>(std::stoull(s));
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
    }
    (void)warm;

    VTUNE_RESUME();
    const auto t0 = high_resolution_clock::now();

    const long long r0 = run_kernel("naive_sum", a.data(), n, repeat);

    const auto t1 = high_resolution_clock::now();
    VTUNE_PAUSE();

    if (r0 != expected) {
        cerr << "wrong result: expected=" << expected
             << ", naive=" << r0 << '\n';
        return 1;
    }

    const double elapsed_ms = duration<double, milli>(t1 - t0).count();
    cout << "mode=naive"
         << ", n=" << n
         << ", repeat=" << repeat
         << ", elapsed_ms=" << elapsed_ms << '\n';

#if HAS_ITT_HEADER
    cout << "ITT task markers enabled for VTune." << '\n';
#else
    cout << "ITT task markers disabled; code still works in VTune, but tasks will not be labeled unless compiled with -DUSE_ITT and linked with ITT." << '\n';
#endif

    cout << "Result is correct." << '\n';
    return 0;
}