#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <string>

using namespace std;
using namespace std::chrono;

// ==============================
// 1. 朴素累加
// ==============================
long long naive_sum(const vector<int>& a) {
    long long sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i];
    }
    return sum;
}

// ==============================
// 2. 两路链式累加
// ==============================
long long way2_sum(const vector<int>& a) {
    long long s0 = 0, s1 = 0;
    size_t i = 0, n = a.size();
    for (; i + 1 < n; i += 2) {
        s0 += a[i];
        s1 += a[i + 1];
    }
    if (i < n) s0 += a[i];
    return s0 + s1;
}

// ==============================
// 3. 四路链式累加
// ==============================
long long way4_sum(const vector<int>& a) {
    long long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0, n = a.size();
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    for (; i < n; ++i) s0 += a[i];
    return s0 + s1 + s2 + s3;
}

// ==============================
// 4. 八路链式累加
// ==============================
long long way8_sum(const vector<int>& a) {
    long long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    long long s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    size_t i = 0, n = a.size();
    for (; i + 7 < n; i += 8) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
        s4 += a[i + 4];
        s5 += a[i + 5];
        s6 += a[i + 6];
        s7 += a[i + 7];
    }
    for (; i < n; ++i) s0 += a[i];
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

// ==============================
// 5. 十六路链式累加
// ==============================
long long way16_sum(const vector<int>& a) {
    long long s[16] = {0};
    size_t i = 0, n = a.size();
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

// ==============================
// 6. 递归二叉归约
// ==============================
long long recursive_pairwise_impl(const vector<int>& a, size_t l, size_t r) {
    if (r - l == 0) return 0;
    if (r - l == 1) return a[l];
    size_t mid = l + (r - l) / 2;
    return recursive_pairwise_impl(a, l, mid) + recursive_pairwise_impl(a, mid, r);
}

long long recursive_pairwise_sum(const vector<int>& a) {
    return recursive_pairwise_impl(a, 0, a.size());
}

// ==============================
// 7. 循环二叉归约
// ==============================
long long iterative_pairwise_sum(const vector<int>& a) {
    if (a.empty()) return 0;

    vector<long long> buf(a.begin(), a.end());
    size_t m = buf.size();

    while (m > 1) {
        size_t half = m / 2;
        for (size_t i = 0; i < half; ++i) {
            buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        if (m % 2 == 1) {
            buf[half] = buf[m - 1];
            m = half + 1;
        } else {
            m = half;
        }
    }
    return buf[0];
}

// ==============================
// 8. 分治算法
// ==============================
long long divide_conquer_impl(const vector<int>& a, size_t l, size_t r) {
    const size_t THRESHOLD = 1024;

    if (r - l <= THRESHOLD) {
        long long sum = 0;
        for (size_t i = l; i < r; ++i) sum += a[i];
        return sum;
    }

    size_t mid = l + (r - l) / 2;
    long long left = divide_conquer_impl(a, l, mid);
    long long right = divide_conquer_impl(a, mid, r);
    return left + right;
}

long long divide_conquer_sum(const vector<int>& a) {
    return divide_conquer_impl(a, 0, a.size());
}

// ==============================
// 9. std::accumulate
// ==============================
long long std_accumulate_sum(const vector<int>& a) {
    return accumulate(a.begin(), a.end(), 0LL);
}

// ==============================
// 自动设置内部重复次数
// ==============================
int get_inner_loop(size_t n) {
    if (n <= (1ull << 12)) return 50000;
    if (n <= (1ull << 14)) return 20000;
    if (n <= (1ull << 16)) return 5000;
    if (n <= (1ull << 18)) return 1000;
    if (n <= (1ull << 20)) return 200;
    if (n <= (1ull << 22)) return 50;
    if (n <= (1ull << 24)) return 10;
    return 3;
}

struct MeasureResult {
    long long result = 0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double time_per_elem_ns = 0.0;
};

template <typename Func>
MeasureResult measure_algorithm(
    const vector<int>& a,
    Func sum_func,
    int warmup,
    int rounds,
    int inner_loop
) {
    volatile long long warm = 0;
    for (int i = 0; i < warmup; ++i) {
        for (int j = 0; j < inner_loop; ++j) {
            warm = sum_func(a);
        }
    }

    vector<double> times_ms;
    long long final_result = 0;

    for (int r = 0; r < rounds; ++r) {
        auto start = high_resolution_clock::now();

        volatile long long tmp = 0;
        for (int j = 0; j < inner_loop; ++j) {
            tmp = sum_func(a);
        }

        auto end = high_resolution_clock::now();
        double elapsed_ms = duration<double, milli>(end - start).count();
        times_ms.push_back(elapsed_ms);
        final_result = tmp;
    }

    double avg_ms = 0.0;
    for (double t : times_ms) avg_ms += t;
    avg_ms /= rounds;

    double min_ms = *min_element(times_ms.begin(), times_ms.end());

    MeasureResult mr;
    mr.result = final_result;
    mr.avg_ms = avg_ms;
    mr.min_ms = min_ms;
    mr.time_per_elem_ns = (avg_ms * 1e6) / (static_cast<double>(a.size()) * inner_loop);
    return mr;
}

// ==============================
// 进度条显示
// ==============================
void print_progress_bar(size_t current, size_t total, size_t n, int inner_loop) {
    const int bar_width = 40;
    double ratio = static_cast<double>(current) / total;
    int filled = static_cast<int>(ratio * bar_width);

    double data_mb_now = (n * sizeof(int)) / (1024.0 * 1024.0);

    cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) cout << "=";
        else if (i == filled) cout << ">";
        else cout << " ";
    }

    cout << "] "
         << fixed << setprecision(1) << (ratio * 100.0) << "% "
         << "(" << current << "/" << total << ")"
         << ", n=" << n
         << ", data=" << setprecision(3) << data_mb_now << "MB"
         << ", inner_loop=" << inner_loop
         << "      " << flush;
}

int main() {
    const int rounds = 10;
    const int warmup = 3;
    const char* output_file = "sum_all_compare_results.csv";

    // ==============================
    // 分段变步长采样 + cache边界加密采样
    // int 为 4 字节
    // ==============================
    vector<size_t> ns;

    // 小规模：细采样
    for (size_t n = 1000; n <= 20000; n += 500) ns.push_back(n);

    // L1/L2附近：中细采样
    for (size_t n = 22000; n <= 100000; n += 2000) ns.push_back(n);

    // 中等规模：中采样
    for (size_t n = 120000; n <= 1000000; n += 20000) ns.push_back(n);

    // L3附近到更大规模：粗采样
    for (size_t n = 1200000; n <= 10000000; n += 200000) ns.push_back(n);

    // 主存区：更粗采样
    for (size_t n = 11000000; n <= 50000000; n += 1000000) ns.push_back(n);

    // cache边界附近额外加密点
    vector<size_t> special_points = {
        6000, 7000, 8000, 8192, 9000, 10000,
        50000, 60000, 65536, 70000, 80000,
        1800000, 2000000, 2097152, 2200000, 2500000
    };

    for (size_t x : special_points) ns.push_back(x);

    sort(ns.begin(), ns.end());
    ns.erase(unique(ns.begin(), ns.end()), ns.end());

    ofstream fout(output_file);
    if (!fout.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        return 1;
    }

    fout
        << "n,data_bytes,data_kb,data_mb,expected,inner_loop,"
        << "naive_result,naive_avg_ms,naive_min_ms,naive_time_per_elem_ns,"
        << "way2_result,way2_avg_ms,way2_min_ms,way2_time_per_elem_ns,way2_speedup_vs_naive,"
        << "way4_result,way4_avg_ms,way4_min_ms,way4_time_per_elem_ns,way4_speedup_vs_naive,"
        << "way8_result,way8_avg_ms,way8_min_ms,way8_time_per_elem_ns,way8_speedup_vs_naive,"
        << "way16_result,way16_avg_ms,way16_min_ms,way16_time_per_elem_ns,way16_speedup_vs_naive,"
        << "recursive_result,recursive_avg_ms,recursive_min_ms,recursive_time_per_elem_ns,recursive_speedup_vs_naive,"
        << "iterative_result,iterative_avg_ms,iterative_min_ms,iterative_time_per_elem_ns,iterative_speedup_vs_naive,"
        << "divide_result,divide_avg_ms,divide_min_ms,divide_time_per_elem_ns,divide_speedup_vs_naive,"
        << "stdacc_result,stdacc_avg_ms,stdacc_min_ms,stdacc_time_per_elem_ns,stdacc_speedup_vs_naive\n";

    cout << "Testing all algorithms...\n";
    cout << "Output CSV: " << output_file << "\n";
    cout << "Total sampling points: " << ns.size() << "\n";

    for (size_t idx = 0; idx < ns.size(); ++idx) {
        size_t n = ns[idx];

        vector<int> a(n, 1);
        long long expected = static_cast<long long>(n);
        int inner_loop = get_inner_loop(n);

        auto naive = measure_algorithm(a, naive_sum, warmup, rounds, inner_loop);
        auto way2 = measure_algorithm(a, way2_sum, warmup, rounds, inner_loop);
        auto way4 = measure_algorithm(a, way4_sum, warmup, rounds, inner_loop);
        auto way8 = measure_algorithm(a, way8_sum, warmup, rounds, inner_loop);
        auto way16 = measure_algorithm(a, way16_sum, warmup, rounds, inner_loop);
        auto recursive = measure_algorithm(a, recursive_pairwise_sum, warmup, rounds, inner_loop);
        auto iterative = measure_algorithm(a, iterative_pairwise_sum, warmup, rounds, inner_loop);
        auto divide = measure_algorithm(a, divide_conquer_sum, warmup, rounds, inner_loop);
        auto stdacc = measure_algorithm(a, std_accumulate_sum, warmup, rounds, inner_loop);

        auto check_result = [&](const string& name, long long got) {
            if (got != expected) {
                cerr << "\n" << name << " result error at n = " << n
                     << ", expected = " << expected
                     << ", got = " << got << endl;
                exit(1);
            }
        };

        check_result("naive", naive.result);
        check_result("way2", way2.result);
        check_result("way4", way4.result);
        check_result("way8", way8.result);
        check_result("way16", way16.result);
        check_result("recursive", recursive.result);
        check_result("iterative", iterative.result);
        check_result("divide", divide.result);
        check_result("stdacc", stdacc.result);

        size_t data_bytes = n * sizeof(int);
        double data_kb = data_bytes / 1024.0;
        double data_mb = data_bytes / (1024.0 * 1024.0);

        auto speedup = [&](double x) -> double {
            return naive.avg_ms / x;
        };

        fout
            << n << ","
            << data_bytes << ","
            << fixed << setprecision(4) << data_kb << ","
            << fixed << setprecision(6) << data_mb << ","
            << expected << ","
            << inner_loop << ","

            << naive.result << ","
            << fixed << setprecision(6) << naive.avg_ms << ","
            << fixed << setprecision(6) << naive.min_ms << ","
            << fixed << setprecision(6) << naive.time_per_elem_ns << ","

            << way2.result << ","
            << fixed << setprecision(6) << way2.avg_ms << ","
            << fixed << setprecision(6) << way2.min_ms << ","
            << fixed << setprecision(6) << way2.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(way2.avg_ms) << ","

            << way4.result << ","
            << fixed << setprecision(6) << way4.avg_ms << ","
            << fixed << setprecision(6) << way4.min_ms << ","
            << fixed << setprecision(6) << way4.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(way4.avg_ms) << ","

            << way8.result << ","
            << fixed << setprecision(6) << way8.avg_ms << ","
            << fixed << setprecision(6) << way8.min_ms << ","
            << fixed << setprecision(6) << way8.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(way8.avg_ms) << ","

            << way16.result << ","
            << fixed << setprecision(6) << way16.avg_ms << ","
            << fixed << setprecision(6) << way16.min_ms << ","
            << fixed << setprecision(6) << way16.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(way16.avg_ms) << ","

            << recursive.result << ","
            << fixed << setprecision(6) << recursive.avg_ms << ","
            << fixed << setprecision(6) << recursive.min_ms << ","
            << fixed << setprecision(6) << recursive.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(recursive.avg_ms) << ","

            << iterative.result << ","
            << fixed << setprecision(6) << iterative.avg_ms << ","
            << fixed << setprecision(6) << iterative.min_ms << ","
            << fixed << setprecision(6) << iterative.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(iterative.avg_ms) << ","

            << divide.result << ","
            << fixed << setprecision(6) << divide.avg_ms << ","
            << fixed << setprecision(6) << divide.min_ms << ","
            << fixed << setprecision(6) << divide.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(divide.avg_ms) << ","

            << stdacc.result << ","
            << fixed << setprecision(6) << stdacc.avg_ms << ","
            << fixed << setprecision(6) << stdacc.min_ms << ","
            << fixed << setprecision(6) << stdacc.time_per_elem_ns << ","
            << fixed << setprecision(6) << speedup(stdacc.avg_ms)
            << "\n";

        print_progress_bar(idx + 1, ns.size(), n, inner_loop);
    }

    cout << endl;
    fout.close();
    cout << "Done. CSV saved to " << output_file << endl;
    return 0;
}