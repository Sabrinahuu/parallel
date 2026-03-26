#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace std::chrono;

volatile double sink = 0.0;

// 初始化矩阵和向量：单块连续内存
void init_data(vector<double>& A, vector<double>& x, int n) {
    A.assign(static_cast<size_t>(n) * n, 0.0);
    x.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        x[i] = (i % 10) + 1;
        for (int j = 0; j < n; j++) {
            A[static_cast<size_t>(i) * n + j] = ((i + j) % 17) + 1;
        }
    }
}

// cache优化算法：逐行访问
// res[j] = sum_i A[i][j] * x[i]
void matrix_column_dot_cache_opt(const vector<double>& A,
                                 const vector<double>& x,
                                 vector<double>& res,
                                 int n) {
    fill(res.begin(), res.end(), 0.0);

    for (int i = 0; i < n; i++) {
        double xi = x[i];
        const double* row = &A[static_cast<size_t>(i) * n];
        for (int j = 0; j < n; j++) {
            res[j] += row[j] * xi;
        }
    }
}

// 根据规模自适应设置 repeat
int get_repeat(int n) {
    if (n <= 32) return 3000;
    else if (n <= 64) return 2000;
    else if (n <= 128) return 1000;
    else if (n <= 256) return 500;
    else if (n <= 512) return 150;
    else return 30;
}

int main() {
    vector<int> sizes;
    for (int n = 1; n <= 2000; n++) {
        sizes.push_back(n);
    }

    int rounds = 3;

    ofstream fout("matrix_cache_opt_dense_results.csv");
    if (!fout.is_open()) {
        cerr << "Failed to open CSV file.\n";
        return 1;
    }

    fout << "n,repeat,round,total_ms,avg_ms,first_result\n";

    cout << fixed << setprecision(6);
    cout << "Matrix Cache Optimized Dense Test (contiguous row-major)\n";
    cout << "n\trepeat\tround\ttotal(ms)\tavg(ms)\tfirst_result\n";

    for (int n : sizes) {
        vector<double> A;
        vector<double> x, res;

        init_data(A, x, n);
        res.resize(n);

        int repeat = get_repeat(n);

        // 热身 3 次
        for (int w = 0; w < 3; w++) {
            matrix_column_dot_cache_opt(A, x, res, n);
            sink += res[0];
        }

        // 正式测试 3 轮
        for (int r = 1; r <= rounds; r++) {
            auto start = high_resolution_clock::now();

            for (int t = 0; t < repeat; t++) {
                matrix_column_dot_cache_opt(A, x, res, n);
                sink += res[0];
            }

            auto end = high_resolution_clock::now();

            double total_ms = duration<double, milli>(end - start).count();
            double avg_ms = total_ms / repeat;

            cout << n << '\t'
                 << repeat << '\t'
                 << r << '\t'
                 << total_ms << '\t'
                 << avg_ms << '\t'
                 << res[0] << '\n';

            fout << n << ','
                 << repeat << ','
                 << r << ','
                 << total_ms << ','
                 << avg_ms << ','
                 << res[0] << '\n';
        }
    }

    fout.close();
    cout << "\nSaved to matrix_cache_opt_dense_results.csv\n";

    return 0;
}