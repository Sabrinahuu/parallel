#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
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
        if (rank == 0) cerr << "Need at least 2 processes!" << endl;
        MPI_Finalize();
        return 1;
    }

    // MD5正确性测试只让rank==0做
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
        for (int i = 0; i < 8; i++)
        {
            bit32 state[4];
            MD5Hash(test_pws[i], state);
            stringstream ss;
            for (int i1 = 0; i1 < 4; i1++)
                ss << setw(8) << setfill('0') << hex << state[i1];
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

    // 所有进程都训练模型
    double time_train = 0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt", rank);
    q.m.order(rank);
    auto end_train = system_clock::now();
    time_train = double(duration_cast<microseconds>(end_train - start_train).count())
                 * microseconds::period::num / microseconds::period::den;

    // ============================================================
    // 流水线设计（生产者-消费者模型）：
    //
    // 进程0（生成进程）：
    //   串行维护优先队列，生成口令
    //   攒够一批后用 MPI_Isend 非阻塞发送给加密进程
    //   发送完立即继续生成，不等待加密进程完成
    //   轮询发送给进程1~(nprocs-1)
    //
    // 进程1~(nprocs-1)（加密进程）：
    //   用 MPI_Recv 阻塞等待接收口令
    //   收到后立即进行MD5哈希
    //   与进程0的生成过程真正并行（流水线）
    //
    // 关键：MPI_Isend使进程0发送不阻塞，生成和加密在时间上重叠
    // ============================================================

    if (rank == 0)
    {
        // -------------------------------------------------------
        // 生成进程
        // -------------------------------------------------------
        q.init();
        cout << "here" << endl;

        double time_guess = 0;
        int curr_num = 0;
        int history  = 0;
        int hash_worker_count = nprocs - 1;
        int next_worker = 1;  // 轮询目标

        // 用于非阻塞发送的请求句柄和缓冲区
        // 需要保存上一次发送的buffer，确保发送完成前buffer不被释放
        MPI_Request send_req = MPI_REQUEST_NULL;
        string send_buf;  // 当前正在发送的buffer

        auto start = system_clock::now();

        while (!q.priority.empty())
        {
            q.PopNext();
            q.total_guesses = q.guesses.size();

            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                if (history + q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    time_guess = double(duration_cast<microseconds>(end - start).count())
                                 * microseconds::period::num / microseconds::period::den;
                    // 等待最后一次发送完成
                    if (send_req != MPI_REQUEST_NULL)
                        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                    cout << "Guess time:" << time_guess << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    break;
                }
            }

            // 攒够一批，发送给加密进程
            if (curr_num > 1000000)
            {
                // 等待上一次非阻塞发送完成，确保send_buf可以被覆盖
                if (send_req != MPI_REQUEST_NULL)
                    MPI_Wait(&send_req, MPI_STATUS_IGNORE);

                // 打包本批次口令
                send_buf.clear();
                for (string &pw : q.guesses)
                    send_buf += pw + "\n";

                int buf_len = send_buf.size();

                // 先同步发送长度（长度很小，开销可忽略）
                MPI_Send(&buf_len, 1, MPI_INT, next_worker, TAG_DATA, MPI_COMM_WORLD);

                // 非阻塞发送口令数据，发送完立即继续生成
                // 进程0不等待加密进程完成，真正实现流水线
                MPI_Isend(send_buf.c_str(), buf_len, MPI_CHAR,
                          next_worker, TAG_DATA, MPI_COMM_WORLD, &send_req);

                // 轮询切换到下一个加密进程
                next_worker = (next_worker % hash_worker_count) + 1;

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // 确保所有发送完成
        if (send_req != MPI_REQUEST_NULL)
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);

        // 向所有加密进程发送退出信号
        for (int i = 1; i < nprocs; i++)
        {
            int exit_flag = 0;
            MPI_Send(&exit_flag, 1, MPI_INT, i, TAG_EXIT, MPI_COMM_WORLD);
        }
    }
    else
    {
        // -------------------------------------------------------
        // 加密进程（进程1 ~ nprocs-1）
        // 与进程0的生成过程并行执行，实现流水线
        // -------------------------------------------------------
        double time_hash = 0;

        while (true)
        {
            // 探测消息类型
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_EXIT)
            {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_EXIT, MPI_COMM_WORLD, &status);
                break;
            }

            // 收到数据：先收长度
            int buf_len;
            MPI_Recv(&buf_len, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, &status);

            // 再收口令数据
            vector<char> buf(buf_len + 1, '\0');
            MPI_Recv(buf.data(), buf_len, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, &status);

            // 立即开始哈希，与进程0的下一轮生成并行
            auto start_hash = system_clock::now();
            bit32 state[4];
            string all_str(buf.begin(), buf.begin() + buf_len);
            size_t pos = 0, found;
            while ((found = all_str.find('\n', pos)) != string::npos)
            {
                string pw = all_str.substr(pos, found - pos);
                if (!pw.empty())
                    MD5Hash(pw, state);
                pos = found + 1;
            }
            auto end_hash = system_clock::now();
            time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                         * microseconds::period::num / microseconds::period::den;
        }

        // 各加密进程输出自己的哈希时间，体现流水线并行效果
        cout << "Rank " << rank << " hash time: " << time_hash << "seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}