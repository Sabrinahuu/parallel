#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <string>
using namespace std;
using namespace chrono;

#define TAG_DATA 1
#define TAG_EXIT 2

int main(int argc, char* argv[])
{
    // MPI+OpenMP混合编程初始化
    // MPI_THREAD_FUNNELED：只有主线程调用MPI函数，OpenMP线程只做计算
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2)
    {
        if (rank == 0) cerr << "Need at least 2 processes!" << endl;
        MPI_Finalize();
        return 1;
    }

    // MD5正确性测试只让rank0做
    if (rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = {
            "123456","password","12345678","qwerty",
            "123456789","12345","1234","111111"
        };
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
        for (int i = 0; i < 8; i++)
        {
            bit32 state[4];
            MD5Hash(test_pws[i], state);
            stringstream ss;
            for (int j = 0; j < 4; j++)
                ss << setw(8) << setfill('0') << hex << state[j];
            if (ss.str() != test_hashes[i])
            {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        cout << "MD5Hash test passed!" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 只让rank0训练模型
    double time_train = 0.0;
    PriorityQueue q;

    if (rank == 0)
    {
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank);
        q.m.order(rank);
        auto end_train = system_clock::now();
        time_train = double(duration_cast<microseconds>(end_train - start_train).count())
                     * microseconds::period::num / microseconds::period::den;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        // -------------------------------------------------------
        // 生成进程：单线程，与之前完全相同
        // -------------------------------------------------------
        q.init();

        const long long MAX_GUESSES  = 10000000;
        const long long REPORT_INTERVAL = 100000;
        const size_t    BATCH_SIZE   = 1000000;

        double time_guess = 0.0;
        long long history     = 0;
        long long last_report = 0;

        int worker_count = nprocs - 1;
        int next_worker  = 1;

        vector<MPI_Request> send_reqs(worker_count, MPI_REQUEST_NULL);
        vector<string>      send_bufs(worker_count);

        auto send_batch = [&](int worker, vector<string>& guesses)
        {
            if (guesses.empty()) return;
            int idx = worker - 1;

            if (send_reqs[idx] != MPI_REQUEST_NULL)
            {
                MPI_Wait(&send_reqs[idx], MPI_STATUS_IGNORE);
                send_reqs[idx] = MPI_REQUEST_NULL;
            }

            send_bufs[idx].clear();
            size_t reserve_size = 0;
            for (const string& pw : guesses) reserve_size += pw.size() + 1;
            send_bufs[idx].reserve(reserve_size);
            for (const string& pw : guesses) { send_bufs[idx] += pw; send_bufs[idx] += '\n'; }

            int buf_len = (int)send_bufs[idx].size();
            MPI_Send(&buf_len, 1, MPI_INT, worker, TAG_DATA, MPI_COMM_WORLD);
            MPI_Isend(send_bufs[idx].c_str(), buf_len, MPI_CHAR,
                      worker, TAG_DATA, MPI_COMM_WORLD, &send_reqs[idx]);

            history += (long long)guesses.size();
            guesses.clear();
        };

        cout << "Start generating guesses..." << endl;
        auto start = system_clock::now();

        while (!q.priority.empty() && history < MAX_GUESSES)
        {
            q.PopNext();
            long long current_total = history + (long long)q.guesses.size();

            while (current_total - last_report >= REPORT_INTERVAL)
            {
                last_report += REPORT_INTERVAL;
                cout << "Guesses generated: " << last_report << endl;
            }

            if (current_total >= MAX_GUESSES)
            {
                size_t keep_num = (size_t)(MAX_GUESSES - history);
                if (q.guesses.size() > keep_num) q.guesses.resize(keep_num);
                send_batch(next_worker, q.guesses);
                break;
            }

            if (q.guesses.size() >= BATCH_SIZE)
            {
                send_batch(next_worker, q.guesses);
                next_worker = (next_worker % worker_count) + 1;
            }
        }

        if (!q.guesses.empty() && history < MAX_GUESSES)
            send_batch(next_worker, q.guesses);

        for (int i = 0; i < worker_count; i++)
            if (send_reqs[i] != MPI_REQUEST_NULL)
                MPI_Wait(&send_reqs[i], MPI_STATUS_IGNORE);

        auto end = system_clock::now();
        time_guess = double(duration_cast<microseconds>(end - start).count())
                     * microseconds::period::num / microseconds::period::den;

        cout << "Total guesses sent: " << history << endl;
        cout << "Guess time:" << time_guess << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;

        for (int i = 1; i < nprocs; i++)
        {
            int exit_flag = 0;
            MPI_Send(&exit_flag, 1, MPI_INT, i, TAG_EXIT, MPI_COMM_WORLD);
        }
    }
    else
    {
        // -------------------------------------------------------
        // 加密进程：MPI接收 + OpenMP并行MD5
        //
        // 每个加密进程开NUM_HASH_THREADS个线程并行做哈希
        // 主线程负责MPI通信（接收数据），OpenMP线程负责计算
        // 这样MPI通信和哈希计算在进程内也实现了一定程度的流水线
        // -------------------------------------------------------
        const int NUM_HASH_THREADS = 4;  // 每个加密进程的线程数，可调整

        double    time_hash       = 0.0;
        long long local_hash_count = 0;

        while (true)
        {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_EXIT)
            {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_EXIT, MPI_COMM_WORLD, &status);
                break;
            }

            // 接收数据长度
            int buf_len = 0;
            MPI_Recv(&buf_len, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, &status);

            // 接收口令数据
            vector<char> buf(buf_len + 1, '\0');
            MPI_Recv(buf.data(), buf_len, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, &status);

            // 解析成vector<string>，方便OpenMP按下标并行
            vector<string> pws;
            pws.reserve(buf_len / 8);  // 预估口令数量
            string all_str(buf.begin(), buf.begin() + buf_len);
            size_t pos = 0, found;
            while ((found = all_str.find('\n', pos)) != string::npos)
            {
                string pw = all_str.substr(pos, found - pos);
                if (!pw.empty()) pws.push_back(pw);
                pos = found + 1;
            }

            // ★ OpenMP并行MD5哈希
            // 每个线程处理自己负责的口令，线程间完全独立无竞争
            // schedule(dynamic, 1000)：动态调度，每次分配1000个口令
            // 适应不同长度口令导致的计算量不均衡
            long long batch_count = 0;
            auto start_hash = system_clock::now();

            #pragma omp parallel for num_threads(NUM_HASH_THREADS) \
                    schedule(dynamic, 1000) \
                    reduction(+:batch_count)
            for (int i = 0; i < (int)pws.size(); i++)
            {
                bit32 state[4];
                MD5Hash(pws[i], state);
                batch_count++;
            }

            auto end_hash = system_clock::now();
            time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                         * microseconds::period::num / microseconds::period::den;
            local_hash_count += batch_count;
        }

        // 输出各加密进程的统计信息
        printf("Rank %d hash count: %lld, hash time: %fseconds\n",
               rank, local_hash_count, time_hash);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}