#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank);  // 传入rank
    q.m.order(rank);                                              // 传入rank
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 只有rank==0加载测试集、维护优先队列、输出结果
    if (rank == 0)
    {
        unordered_set<std::string> test_set;
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while (test_data >> pw)
        {
            test_count += 1;
            test_set.insert(pw);
            if (test_count >= 1000000) break;
        }
        int cracked = 0;

        q.init();
        cout << "here" << endl;
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;

        while (!q.priority.empty())
        {
            // 通知所有worker进程：即将执行一次Generate
            int flag = 1;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

            q.PopNext();
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
                    cout << "Cracked:" << cracked << endl;
                    break;
                }
            }

            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    if (test_set.find(pw) != test_set.end()) cracked += 1;
                    MD5Hash(pw, state);
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // 循环结束，通知所有worker退出
        int flag = 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        // worker进程：等待rank==0的信号，协助执行Generate
        // Generate内部通过MPI_Gather与rank==0同步
        // 但Generate需要PT信息——见下方说明
        while (true)
        {
            int flag;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (flag == 0) break;
            // worker参与Generate的MPI_Gather
            // 需要rank==0广播PT给workers（见guessing.cpp的修改）
            q.GenerateWorker();  // 新增函数，见下方
        }
    }

    MPI_Finalize();
    return 0;
}