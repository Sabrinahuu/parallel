#include "PCFG.h"
#include <mpi.h>
#include <cstring>
using namespace std;

// 在guessing.cpp顶部加一个阈值
#define MIN_PARALLEL_SIZE 1000  // value数少于这个值就串行处理

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

// 原始串行PopNext，保留不变
void PriorityQueue::PopNext()
{
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

// ============================================================
// 新增：PopNextN
//
// 并行策略：
//   一次从优先队列头部取出 n 个PT（n = 进程数），
//   进程0将第i个PT分配给进程i，各进程独立调用GenerateSingle()
//   生成自己负责的PT的全部猜测，最后Gather到进程0。
//   各PT生成完毕后，收集所有新PT统一插回优先队列。
//
// 与之前方案的区别：
//   之前：1个PT内部的value循环拆分给多进程（数据并行）
//   现在：多个PT分别交给不同进程（任务并行），PT级别的并行
//   两种方案可以叠加，但本函数聚焦PT级别的并行
// ============================================================
void PriorityQueue::PopNextN()
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int batch_size = 0;

    // Step 1: 广播batch_size
    if (rank == 0)
        batch_size = min((int)nprocs, (int)priority.size());
    MPI_Bcast(&batch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    string local_buf = "";

    // Step 2: 每个PT广播信息，进程i处理第i个PT
    for (int i = 0; i < batch_size; i++)
    {
        int prefix_len = 0;
        char prefix_buf[4096] = {0};
        int seg_type = 0, seg_len = 0, total = 0;

        if (rank == 0)
        {
            PT &pt = priority[i];
            CalProb(pt);

            string guess = "";
            if (pt.content.size() > 1)
            {
                int seg_idx = 0;
                for (int idx : pt.curr_indices)
                {
                    if (pt.content[seg_idx].type == 1)
                        guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                    if (pt.content[seg_idx].type == 2)
                        guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                    if (pt.content[seg_idx].type == 3)
                        guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                    seg_idx++;
                    if (seg_idx == (int)pt.content.size() - 1) break;
                }
            }
            prefix_len = guess.size();
            strncpy(prefix_buf, guess.c_str(), sizeof(prefix_buf)-1);

            int last = pt.content.size() - 1;
            seg_type = pt.content[last].type;
            seg_len  = pt.content[last].length;
            total    = pt.max_indices[last];
        }

        MPI_Bcast(&prefix_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(prefix_buf, 4096, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seg_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seg_len,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&total,    1, MPI_INT, 0, MPI_COMM_WORLD);

        // ★ 关键优化：只有total足够大才并行，否则只让进程0处理
        if (total >= MIN_PARALLEL_SIZE)
        {
            if (rank == i)
            {
                string guess(prefix_buf, prefix_len);
                segment tmp(seg_type, seg_len);
                segment *a = nullptr;
                if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
                if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
                if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];
                for (int j = 0; j < total; j++)
                    local_buf += guess + a->ordered_values[j] + "\n";
            }
        }
        else
        {
            // 小PT只让rank==0串行处理，其他进程跳过
            if (rank == 0)
            {
                string guess(prefix_buf, prefix_len);
                segment tmp(seg_type, seg_len);
                segment *a = nullptr;
                if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
                if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
                if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];
                for (int j = 0; j < total; j++)
                    local_buf += guess + a->ordered_values[j] + "\n";
            }
        }
    }

    // Step 3: Gather
    int local_len = local_buf.size();
    vector<int> all_lens(nprocs, 0);
    MPI_Gather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(nprocs, 0);
    int total_recv = 0;
    if (rank == 0)
        for (int i = 0; i < nprocs; i++) { displs[i] = total_recv; total_recv += all_lens[i]; }

    vector<char> recv_buf(max(total_recv, 1));
    MPI_Gatherv(local_buf.c_str(), local_len, MPI_CHAR,
                recv_buf.data(), all_lens.data(), displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        string all_str(recv_buf.begin(), recv_buf.begin() + total_recv);
        size_t pos = 0, found;
        while ((found = all_str.find('\n', pos)) != string::npos)
        {
            string g = all_str.substr(pos, found - pos);
            if (!g.empty()) { guesses.emplace_back(g); total_guesses++; }
            pos = found + 1;
        }
    }

    // Step 4: 新PT插回队列
    if (rank == 0)
    {
        vector<PT> all_new_pts;
        for (int i = 0; i < batch_size; i++)
        {
            vector<PT> new_pts = priority[i].NewPTs();
            for (PT &pt : new_pts) { CalProb(pt); all_new_pts.push_back(pt); }
        }
        priority.erase(priority.begin(), priority.begin() + batch_size);
        for (PT &pt : all_new_pts)
        {
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                if (iter == priority.begin() && iter->prob < pt.prob)
                    { priority.emplace(iter, pt); inserted = true; break; }
                if (iter != priority.end() - 1 && pt.prob <= iter->prob && pt.prob > (iter+1)->prob)
                    { priority.emplace(iter+1, pt); inserted = true; break; }
            }
            if (!inserted) priority.emplace_back(pt);
        }
    }
}
vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
        return res;
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < (int)curr_indices.size() - 1; i++)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// 原始串行Generate，保留
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);
    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        for (int i = 0; i < pt.max_indices[0]; i++)
        {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses++;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }
        segment *a;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
        if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        if (pt.content[last].type == 3) a = &m.symbols[m.FindSymbol(pt.content[last])];
        for (int i = 0; i < pt.max_indices[last]; i++)
        {
            guesses.emplace_back(guess + a->ordered_values[i]);
            total_guesses++;
        }
    }
}

void PriorityQueue::PopNextN_worker()
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 接收batch_size
    int batch_size = 0;
    MPI_Bcast(&batch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    string local_buf = "";

    for (int i = 0; i < batch_size; i++)
    {
        // 接收第i个PT的信息（与PopNextN的广播顺序完全对齐）
        int prefix_len = 0;
        char prefix_buf[4096] = {0};
        int seg_type = 0, seg_len = 0, total = 0;

        MPI_Bcast(&prefix_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(prefix_buf, 4096, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seg_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seg_len,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&total,    1, MPI_INT, 0, MPI_COMM_WORLD);

        // 只有进程i处理第i个PT
        if (rank == i)
        {
            string guess(prefix_buf, prefix_len);
            segment tmp(seg_type, seg_len);
            segment *a = nullptr;
            if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
            if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
            if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];

            for (int j = 0; j < total; j++)
                local_buf += guess + a->ordered_values[j] + "\n";
        }
    }

    // 参与Gather（worker不需要接收结果）
    int local_len = local_buf.size();
    vector<int> all_lens(nprocs, 0);
    MPI_Gather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(nprocs, 0);
    MPI_Gatherv(local_buf.c_str(), local_len, MPI_CHAR,
                nullptr, all_lens.data(), displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);
}