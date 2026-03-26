#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <iomanip>

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
        double xi = x[i];
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
// 4) 循环展开版：flat vector + row access + unroll4
//    每次处理4个元素，减少循环控制开销
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

        // 处理剩余不足4个的元素
        for (; j < n; j++) {
            res[j] += A[row_base + j] * xi;
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
        int row_base = i * n;
        for (int j = 0; j < n; j++) {
            A[row_base + j] = dist(rng);
        }
    }
}

// ============================================================
// 工具函数：结果校验
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

    if (argc >= 2) n = stoi(argv[1]);
    if (argc >= 3) repeat = stoi(argv[2]);

    cout << "Matrix size n = " << n << "\n";
    cout << "Repeat count  = " << repeat << "\n\n";

    // ----------------------------
    // 数据：vector<vector<double>>
    // ----------------------------
    vector<vector<double>> A_vv;
    vector<double> x_vv;
    init_matrix_vectorvector(A_vv, x_vv, n);

    // ----------------------------
    // 数据：连续一维 vector<double>
    // ----------------------------
    vector<double> A_flat;
    vector<double> x_flat;
    init_matrix_flat(A_flat, x_flat, n);

    vector<double> res_bad, res_cache_opt, res_vector_opt, res_unroll4_opt;

    // 1) bad 版
    double t_bad = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_bad(A_vv, x_vv, out, n);
        },
        repeat, res_bad
    );

    // 2) cache_opt 版
    double t_cache_opt = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_cache_opt(A_vv, x_vv, out, n);
        },
        repeat, res_cache_opt
    );

    // 3) vector_opt 版
    double t_vector_opt = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_vector_opt(A_flat, x_flat, out, n);
        },
        repeat, res_vector_opt
    );

    // 4) unroll4_opt 版
    double t_unroll4_opt = benchmark(
        [&](vector<double>& out) {
            matrix_column_dot_unroll4(A_flat, x_flat, out, n);
        },
        repeat, res_unroll4_opt
    );

    cout << fixed << setprecision(6);

    cout << "==== Timing Result ====\n";
    cout << "bad (vector<vector> + column access):           " << t_bad << " s\n";
    cout << "cache_opt (vector<vector> + row access):        " << t_cache_opt << " s\n";
    cout << "vector_opt (flat vector + row access):          " << t_vector_opt << " s\n";
    cout << "unroll4_opt (flat vector + row access + unroll): " << t_unroll4_opt << " s\n\n";

    cout << "==== Checksum ====\n";
    cout << "bad checksum         = " << checksum(res_bad) << "\n";
    cout << "cache_opt checksum   = " << checksum(res_cache_opt) << "\n";
    cout << "vector_opt checksum  = " << checksum(res_vector_opt) << "\n";
    cout << "unroll4_opt checksum = " << checksum(res_unroll4_opt) << "\n\n";

    cout << "==== Suggested version ====\n";
    cout << "Compare vector_opt and unroll4_opt in VTune.\n";

    return 0;
}