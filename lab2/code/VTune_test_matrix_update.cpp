#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace std;
using clk = chrono::high_resolution_clock;

// ============================================================
// 1) 原始差版本：vector<vector<double>> + 按列访问
// ============================================================
void matrix_column_dot_bad(const vector<vector<double>>& A,
                           const vector<double>& x,
                           vector<double>& res,
                           int n) {
    res.assign(n, 0.0);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            res[j] += A[i][j] * x[i];
        }
    }
}

// ============================================================
// 2) cache 优化版：vector<vector<double>> + 按行访问
// ============================================================
void matrix_column_dot_cache_opt(const vector<vector<double>>& A,
                                 const vector<double>& x,
                                 vector<double>& res,
                                 int n) {
    res.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        for (int j = 0; j < n; j++) {
            res[j] += A[i][j] * xi;
        }
    }
}

// ============================================================
// 3) 连续一维 vector 存矩阵：基础优化版
// ============================================================
void matrix_column_dot_vector_opt(const vector<double>& A,
                                  const vector<double>& x,
                                  vector<double>& res,
                                  int n) {
    res.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const int row_base = i * n;
        for (int j = 0; j < n; j++) {
            res[j] += A[row_base + j] * xi;
        }
    }
}

// ============================================================
// 4) 4路展开版
// ============================================================
void matrix_column_dot_unroll4(const vector<double>& A,
                               const vector<double>& x,
                               vector<double>& res,
                               int n) {
    res.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const int row_base = i * n;

        int j = 0;
        for (; j + 3 < n; j += 4) {
            res[j]     += A[row_base + j]     * xi;
            res[j + 1] += A[row_base + j + 1] * xi;
            res[j + 2] += A[row_base + j + 2] * xi;
            res[j + 3] += A[row_base + j + 3] * xi;
        }

        for (; j < n; j++) {
            res[j] += A[row_base + j] * xi;
        }
    }
}

// ============================================================
// 5) 4路展开 + 局部寄存器缓存
// ============================================================
void matrix_column_dot_unroll4_reg(const vector<double>& A,
                                   const vector<double>& x,
                                   vector<double>& res,
                                   int n) {
    res.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const int row_base = i * n;

        int j = 0;
        for (; j + 3 < n; j += 4) {
            double r0 = res[j];
            double r1 = res[j + 1];
            double r2 = res[j + 2];
            double r3 = res[j + 3];

            r0 += A[row_base + j]     * xi;
            r1 += A[row_base + j + 1] * xi;
            r2 += A[row_base + j + 2] * xi;
            r3 += A[row_base + j + 3] * xi;

            res[j]     = r0;
            res[j + 1] = r1;
            res[j + 2] = r2;
            res[j + 3] = r3;
        }

        for (; j < n; j++) {
            res[j] += A[row_base + j] * xi;
        }
    }
}

// ============================================================
// 6) 外层2路展开
// ============================================================
void matrix_column_dot_outer_unroll2(const vector<double>& A,
                                     const vector<double>& x,
                                     vector<double>& res,
                                     int n) {
    res.assign(n, 0.0);

    int i = 0;
    for (; i + 1 < n; i += 2) {
        const double x0 = x[i];
        const double x1 = x[i + 1];
        const int row0 = i * n;
        const int row1 = (i + 1) * n;

        for (int j = 0; j < n; j++) {
            res[j] += A[row0 + j] * x0 + A[row1 + j] * x1;
        }
    }

    for (; i < n; i++) {
        const double xi = x[i];
        const int row = i * n;
        for (int j = 0; j < n; j++) {
            res[j] += A[row + j] * xi;
        }
    }
}

// ============================================================
// 7) 分块版本：按 j 方向 blocking
// ============================================================
void matrix_column_dot_blocked(const vector<double>& A,
                               const vector<double>& x,
                               vector<double>& res,
                               int n,
                               int block_size = 64) {
    res.assign(n, 0.0);

    for (int jb = 0; jb < n; jb += block_size) {
        const int jend = min(jb + block_size, n);

        for (int i = 0; i < n; i++) {
            const double xi = x[i];
            const int row_base = i * n;

            for (int j = jb; j < jend; j++) {
                res[j] += A[row_base + j] * xi;
            }
        }
    }
}

// ============================================================
// 8) 分块 + 4路展开 + 寄存器缓存
// ============================================================
void matrix_column_dot_blocked_unroll4_reg(const vector<double>& A,
                                           const vector<double>& x,
                                           vector<double>& res,
                                           int n,
                                           int block_size = 64) {
    res.assign(n, 0.0);

    for (int jb = 0; jb < n; jb += block_size) {
        const int jend = min(jb + block_size, n);

        for (int i = 0; i < n; i++) {
            const double xi = x[i];
            const int row_base = i * n;

            int j = jb;
            for (; j + 3 < jend; j += 4) {
                double r0 = res[j];
                double r1 = res[j + 1];
                double r2 = res[j + 2];
                double r3 = res[j + 3];

                r0 += A[row_base + j]     * xi;
                r1 += A[row_base + j + 1] * xi;
                r2 += A[row_base + j + 2] * xi;
                r3 += A[row_base + j + 3] * xi;

                res[j]     = r0;
                res[j + 1] = r1;
                res[j + 2] = r2;
                res[j + 3] = r3;
            }

            for (; j < jend; j++) {
                res[j] += A[row_base + j] * xi;
            }
        }
    }
}

// ============================================================
// 工具函数：初始化数据
// ============================================================
void init_matrix_vectorvector(vector<vector<double>>& A,
                              vector<double>& x,
                              int n,
                              unsigned seed = 42) {
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);

    A.assign(n, vector<double>(n));
    x.resize(n);

    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        for (int j = 0; j < n; j++) {
            A[i][j] = dist(rng);
        }
    }
}

void init_matrix_flat(vector<double>& A,
                      vector<double>& x,
                      int n,
                      unsigned seed = 42) {
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);

    A.resize(n * n);
    x.resize(n);

    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        const int row_base = i * n;
        for (int j = 0; j < n; j++) {
            A[row_base + j] = dist(rng);
        }
    }
}

// ============================================================
// 结果校验
// ============================================================
double checksum(const vector<double>& v) {
    double s = 0.0;
    for (double e : v) s += e;
    return s;
}

// ============================================================
// 计时函数
// ============================================================
template <typename Func>
double benchmark(Func fn, int repeat, vector<double>& out_res) {
    auto start = clk::now();
    for (int t = 0; t < repeat; t++) {
        fn(out_res);
    }
    auto end = clk::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

int main(int argc, char** argv) {
    int n = 2048;
    int repeat = 20;
    int block_size = 64;

    if (argc >= 2) n = stoi(argv[1]);
    if (argc >= 3) repeat = stoi(argv[2]);
    if (argc >= 4) block_size = stoi(argv[3]);

    cout << "Matrix size n = " << n << "\n";
    cout << "Repeat count  = " << repeat << "\n";
    cout << "Block size    = " << block_size << "\n\n";

    // vector<vector<double>> 数据
    vector<vector<double>> A_vv;
    vector<double> x_vv;
    init_matrix_vectorvector(A_vv, x_vv, n);

    // flat vector 数据
    vector<double> A_flat;
    vector<double> x_flat;
    init_matrix_flat(A_flat, x_flat, n);

    vector<double> res_bad, res_cache_opt, res_vector_opt;
    vector<double> res_unroll4, res_unroll4_reg, res_outer_unroll2;
    vector<double> res_blocked, res_blocked_unroll4_reg;

    double t_bad = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_bad(A_vv, x_vv, out, n);
        },
        repeat, res_bad
    );

    double t_cache_opt = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_cache_opt(A_vv, x_vv, out, n);
        },
        repeat, res_cache_opt
    );

    double t_vector_opt = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_vector_opt(A_flat, x_flat, out, n);
        },
        repeat, res_vector_opt
    );

    double t_unroll4 = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_unroll4(A_flat, x_flat, out, n);
        },
        repeat, res_unroll4
    );

    double t_unroll4_reg = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_unroll4_reg(A_flat, x_flat, out, n);
        },
        repeat, res_unroll4_reg
    );

    double t_outer_unroll2 = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_outer_unroll2(A_flat, x_flat, out, n);
        },
        repeat, res_outer_unroll2
    );

    double t_blocked = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_blocked(A_flat, x_flat, out, n, block_size);
        },
        repeat, res_blocked
    );

    double t_blocked_unroll4_reg = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_blocked_unroll4_reg(A_flat, x_flat, out, n, block_size);
        },
        repeat, res_blocked_unroll4_reg
    );

    cout << fixed << setprecision(6);

    cout << "==== Timing Result ====\n";
    cout << "bad                         : " << t_bad << " s\n";
    cout << "cache_opt                   : " << t_cache_opt << " s\n";
    cout << "vector_opt                  : " << t_vector_opt << " s\n";
    cout << "unroll4                     : " << t_unroll4 << " s\n";
    cout << "unroll4_reg                 : " << t_unroll4_reg << " s\n";
    cout << "outer_unroll2               : " << t_outer_unroll2 << " s\n";
    cout << "blocked                     : " << t_blocked << " s\n";
    cout << "blocked_unroll4_reg         : " << t_blocked_unroll4_reg << " s\n\n";

    cout << "==== Checksum ====\n";
    cout << "bad                         = " << checksum(res_bad) << "\n";
    cout << "cache_opt                   = " << checksum(res_cache_opt) << "\n";
    cout << "vector_opt                  = " << checksum(res_vector_opt) << "\n";
    cout << "unroll4                     = " << checksum(res_unroll4) << "\n";
    cout << "unroll4_reg                 = " << checksum(res_unroll4_reg) << "\n";
    cout << "outer_unroll2               = " << checksum(res_outer_unroll2) << "\n";
    cout << "blocked                     = " << checksum(res_blocked) << "\n";
    cout << "blocked_unroll4_reg         = " << checksum(res_blocked_unroll4_reg) << "\n\n";

    cout << "==== Note ====\n";
    cout << "Use VTune to compare CPU Time / Memory Access / Microarchitecture.\n";

    return 0;
}