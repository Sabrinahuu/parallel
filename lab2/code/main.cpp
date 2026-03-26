#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

volatile double sink = 0.0;

void init_data(vector<vector<double>>& A, vector<double>& x, int n) {
    A.assign(n, vector<double>(n));
    x.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        x[i] = (i % 10) + 1;
        for (int j = 0; j < n; j++) {
            A[i][j] = ((i + j) % 17) + 1;
        }
    }
}

// 平凡算法：逐列访问
void matrix_column_dot_naive(const vector<vector<double>>& A,
                             const vector<double>& x,
                             vector<double>& res,
                             int n) {
    res.assign(n, 0.0);
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[i][j] * x[i];
        }
        res[j] = sum;
    }
}

int main() {
    vector<int> sizes = {64, 128, 256, 512, 1024};

    cout << fixed << setprecision(4);
    cout << "Matrix Naive (column-wise access)\n";
    cout << "n\trepeat\ttime(ms)\tfirst_result\n";

    for (int n : sizes) {
        vector<vector<double>> A;
        vector<double> x, res;
        init_data(A, x, n);

        int repeat;
        if (n <= 128) repeat = 500;
        else if (n <= 256) repeat = 200;
        else if (n <= 512) repeat = 50;
        else repeat = 10;

        auto start = high_resolution_clock::now();
        for (int t = 0; t < repeat; t++) {
            matrix_column_dot_naive(A, x, res, n);
            sink += res[0];
        }
        auto end = high_resolution_clock::now();

        double ms = duration<double, milli>(end - start).count();
        cout << n << '\t' << repeat << '\t' << ms << '\t' << res[0] << '\n';
    }

    return 0;
}