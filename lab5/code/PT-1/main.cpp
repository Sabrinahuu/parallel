#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // MD5测试只让rank==0做
    if (rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
        string test_hashes[8] = {
            "e10adc3949ba59abbe56e057f20f883e",
            "5f4dcc3b5aa765d61d8327deb882cf99",
            "25d55ad283aa400af464c76d713c07ad",
            "d8578edf8458ce06fbc5bb76a58c5ca4",
            "25f9e794323b453885f5181f1b624d0b",
            "827ccb0eea8a706c4c34a16891f84e7b",
            "81dc9bdb52d04dc20036dbd8313ed055",
            "96e79218965eb72c92a549dd5a330112"
        };
        for (int i = 0; i < 8; i++) {
            bit32 state[4];
            MD5Hash(test_pws[i], state);
            stringstream ss;
            for (int i1 = 0; i1 < 4; i1 += 1)
                ss << std::setw(8) << std::setfill('0') << hex << state[i1];
            if (ss.str() != test_hashes[i]) {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);  // 失败时终止所有进程
                return 1;
            }
        }
        cout << "MD5Hash test passed!" << endl;
    }
    // 等待rank==0的MD5测试完成
    MPI_Barrier(MPI_COMM_WORLD);

    // 所有进程都训练（rank控制输出）
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank);
    q.m.order(rank);
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    if (rank == 0)
    {
        q.init();
        cout << "here" << endl;
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;

        while (!q.priority.empty())
        {
            int flag = 1;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

            q.PopNextN();
            q.total_guesses = q.guesses.size();

            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                if (history + q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    break;
                }
            }

            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : q.guesses)
                    MD5Hash(pw, state);
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // 通知workers退出
        int flag = 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        while (true)
        {
            int flag;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (flag == 0) break;
            q.PopNextN_worker();  // 只参与通信，不操作队列
        }
    }

    MPI_Finalize();
    return 0;
}