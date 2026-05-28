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

    // MD5正确性测试，只让进程0做
    if (rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = {"123456","password","12345678","qwerty",
                               "123456789","12345","1234","111111"};
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
            for (int i1 = 0; i1 < 4; i1++)
                ss << setw(8) << setfill('0') << hex << state[i1];
            if (ss.str() != test_hashes[i]) {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        cout << "MD5Hash test passed!" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 所有进程都训练，rank控制输出
    double time_hash = 0, time_guess = 0, time_train = 0;
    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank);
    q.m.order(rank);
    auto end_train = system_clock::now();
    time_train = double(duration_cast<microseconds>(end_train - start_train).count())
                 * microseconds::period::num / microseconds::period::den;

    // 所有进程都init，每个进程都有完整的模型和队列
    q.init();
    if (rank == 0) cout << "here" << endl;

    int curr_num = 0;
    int history  = 0;
    auto start = system_clock::now();

    while (true)
    {
        // 进程0检查队列，广播是否继续
        int global_has_work = 0;
        if (rank == 0)
            global_has_work = q.priority.empty() ? 0 : 1;
        MPI_Bcast(&global_has_work, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!global_has_work) break;

        // 所有进程参与批量生成
        q.PopNextBatch();

        // 统计全局猜测总数
        int local_count = (int)q.guesses.size();
        int global_count = 0;
        MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = global_count;

        int should_exit = 0;
        if (rank == 0)
        {
            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;
                if (history + q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    time_guess = double(duration_cast<microseconds>(end - start).count())
                                 * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    should_exit = 1;
                }
            }
            // 进程0内存管理
            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string &pw : q.guesses)
                    MD5Hash(pw, state);
                auto end_hash = system_clock::now();
                time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                             * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
        else
        {
            // 其他进程也需要定期清空guesses避免内存溢出
            if ((int)q.guesses.size() > 1000000)
            {
                bit32 state[4];
                for (string &pw : q.guesses)
                    MD5Hash(pw, state);
                q.guesses.clear();
            }
        }

        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (should_exit) break;
    }

    MPI_Finalize();
    return 0;
}