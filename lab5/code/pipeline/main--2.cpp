#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <string>
using namespace std;
using namespace chrono;

#define TAG_DATA 1
#define TAG_EXIT 2

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2)
    {
        if (rank == 0)
            cerr << "Need at least 2 processes!" << endl;
        MPI_Finalize();
        return 1;
    }

    // ============================================================
    // 1. MD5 正确性测试：只让rank0执行
    // ============================================================
    if (rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;

        string test_pws[8] = {
            "123456", "password", "12345678", "qwerty",
            "123456789", "12345", "1234", "111111"
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
                MPI_Finalize();
                return 1;
            }
        }

        cout << "MD5Hash test passed!" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================================
    // 2. 只让rank0训练PCFG模型
    // ============================================================
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

    // 保证rank0训练完成后，worker再进入接收状态
    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================================
    // 3. 流水线设计：rank 0 生成口令，其余 rank 计算 MD5
    // ============================================================

    if (rank == 0)
    {
        q.init();

        const long long MAX_GUESSES = 10000000;
        const long long REPORT_INTERVAL = 100000;
        const size_t BATCH_SIZE = 1000000;

        double time_guess = 0.0;
        long long history = 0;       // 已经发送给worker的口令数量
        long long last_report = 0;   // 上一次输出进度的位置

        int worker_count = nprocs - 1;
        int next_worker = 1;

        //每个worker独立保存一个非阻塞发送请求
        vector<MPI_Request> send_reqs(worker_count, MPI_REQUEST_NULL);
        //每个worker独立保存一个发送缓冲区
        vector<string> send_bufs(worker_count);

        auto send_batch = [&](int worker, vector<string>& guesses)
        {
            if (guesses.empty())
                return;

            int idx = worker - 1;

            // 复用worker的发送缓冲区前，确认上一批的MPI_Isend已经真正完成
            if (send_reqs[idx] != MPI_REQUEST_NULL)
            {
                MPI_Wait(&send_reqs[idx], MPI_STATUS_IGNORE);
                send_reqs[idx] = MPI_REQUEST_NULL;
            }

            send_bufs[idx].clear();

            // 预估本批数据需要的空间，减少string自动扩容开销
            size_t reserve_size = 0;
            for (const string& pw : guesses)
                reserve_size += pw.size() + 1;
            send_bufs[idx].reserve(reserve_size);

            for (const string& pw : guesses)
            {
                send_bufs[idx] += pw;
                send_bufs[idx] += '\n';
            }
            
            
            int buf_len = static_cast<int>(send_bufs[idx].size());

            //先发送数据长度，worker收到长度后，再创建合适大小的接收缓冲区
            MPI_Send(&buf_len, 1, MPI_INT, worker, TAG_DATA, MPI_COMM_WORLD);
            
            //再非阻塞发送真正的口令数据
            MPI_Isend(send_bufs[idx].c_str(), buf_len, MPI_CHAR,
                      worker, TAG_DATA, MPI_COMM_WORLD, &send_reqs[idx]);

            history += static_cast<long long>(guesses.size());
            guesses.clear();
        };

        cout << "Start generating guesses..." << endl;
        auto start = system_clock::now();

        while (!q.priority.empty() && history < MAX_GUESSES)
        {
            q.PopNext();

            long long current_total = history + static_cast<long long>(q.guesses.size());

            //// 每生成约 10 万条口令，输出一次进度
            while (current_total - last_report >= REPORT_INTERVAL)
            {
                last_report += REPORT_INTERVAL;
                cout << "Guesses generated: " << last_report << endl;
            }

            //如果本轮生成后超过最大测试数量，
            // 就截断最后一批，避免明显超过 MAX_GUESSES
            if (current_total >= MAX_GUESSES)
            {
                size_t keep_num = static_cast<size_t>(MAX_GUESSES - history);
                if (q.guesses.size() > keep_num)
                    q.guesses.resize(keep_num);

                send_batch(next_worker, q.guesses);
                break;
            }

            // 凑够一批后，将这批口令发送给当前 worker
            if (q.guesses.size() >= BATCH_SIZE)
            {
                send_batch(next_worker, q.guesses);

                // 轮询切换到下一个 worker
                next_worker = (next_worker % worker_count) + 1;
            }
        }

        if (!q.guesses.empty() && history < MAX_GUESSES)
        {
            send_batch(next_worker, q.guesses);
        }

        // 等待所有非阻塞发送完成
        for (int i = 0; i < worker_count; i++)
        {
            if (send_reqs[i] != MPI_REQUEST_NULL)
            {
                MPI_Wait(&send_reqs[i], MPI_STATUS_IGNORE);
                send_reqs[i] = MPI_REQUEST_NULL;
            }
        }

        auto end = system_clock::now();
        time_guess = double(duration_cast<microseconds>(end - start).count())
                     * microseconds::period::num / microseconds::period::den;

        cout << "Total guesses sent: " << history << endl;
        cout << "Guess time:" << time_guess << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;

        //向所有 worker 发送退出信号
        for (int i = 1; i < nprocs; i++)
        {
            int exit_flag = 0;
            MPI_Send(&exit_flag, 1, MPI_INT, i, TAG_EXIT, MPI_COMM_WORLD);
        }
    }
    else
    {
        double time_hash = 0.0;
        long long local_hash_count = 0;

        while (true)
        {
            MPI_Status status;

            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            //如果收到退出消息，则接收该消息并结束循环
            if (status.MPI_TAG == TAG_EXIT)
            {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_EXIT, MPI_COMM_WORLD, &status);
                break;
            }
            
            //否则说明收到的是数据消息
            //第一步：接收本批数据长度
            int buf_len = 0;
            MPI_Recv(&buf_len, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
            
            //第二步：根据长度创建缓冲区，并接收实际口令数据
            vector<char> buf(buf_len + 1, '\0');
            MPI_Recv(buf.data(), buf_len, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, &status);

            auto start_hash = system_clock::now();

            bit32 state[4];
            string all_str(buf.begin(), buf.begin() + buf_len);

            size_t pos = 0;
            size_t found = 0;

            while ((found = all_str.find('\n', pos)) != string::npos)
            {
                string pw = all_str.substr(pos, found - pos);
                if (!pw.empty())
                {
                    MD5Hash(pw, state);
                    local_hash_count++;
                }
                pos = found + 1;
            }

            auto end_hash = system_clock::now();
            time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                         * microseconds::period::num / microseconds::period::den;
        }

        cout << "Rank " << rank
             << " hash count: " << local_hash_count
             << ", hash time: " << time_hash << "seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}