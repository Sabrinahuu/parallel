#include "PCFG.h"
#include <mpi.h>
using namespace std;

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

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
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
    return res;
}

// ============================================================
// MPI并行化的Generate函数
//
// 并行策略：
//   对于每个PT，最后一个segment有N个value需要遍历。
//   将这N个value按进程数平均分配，每个进程负责一段区间 [start, end)。
//   各进程独立生成自己负责的那段猜测，最后由进程0收集汇总。
//
// 为什么只有最后一个segment可以并行？
//   其他segment已经被curr_indices固定为具体值（prefix已确定），
//   只有最后一个segment的遍历是独立重复的工作，天然适合数据并行。
//
// 为什么不对整个优先队列并行？
//   优先队列是有序的串行数据结构，每次PopNext()依赖队首元素，
//   无法简单地让多进程同时操作不同PT（会破坏概率排序语义）。
// ============================================================
// Generate函数修改：rank==0广播PT的关键信息给workers
void PriorityQueue::Generate(PT pt)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    CalProb(pt);

    // 构建prefix（多segment情况）
    string guess = "";
    segment *a;
    int total;

    if (pt.content.size() == 1)
    {
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        total = pt.max_indices[0];
    }
    else
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
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
        if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        if (pt.content[last].type == 3) a = &m.symbols[m.FindSymbol(pt.content[last])];
        total = pt.max_indices[last];
    }

    // 广播prefix字符串长度和内容给所有worker
    int prefix_len = guess.size();
    MPI_Bcast(&prefix_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (prefix_len > 0)
    {
        char* buf = new char[prefix_len + 1];
        if (rank == 0) strcpy(buf, guess.c_str());
        MPI_Bcast(buf, prefix_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) guess = string(buf);
        delete[] buf;
    }

    // 广播total
    MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播segment类型和长度，让worker自己找到对应的segment
    int seg_type = a->type;
    int seg_len  = a->length;
    MPI_Bcast(&seg_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seg_len,  1, MPI_INT, 0, MPI_COMM_WORLD);

    // worker根据type/length找到自己模型里对应的segment
    if (rank != 0)
    {
        segment tmp(seg_type, seg_len);
        if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
        if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
        if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];
    }

    // 每个进程负责自己的区间
    int chunk = total / nprocs;
    int start = rank * chunk;
    int end   = (rank == nprocs - 1) ? total : start + chunk;

    string local_buf;
    for (int i = start; i < end; i++)
        local_buf += guess + a->ordered_values[i] + "\n";

    // Gather到rank==0
    int local_len = local_buf.size();
    vector<int> all_lens(nprocs), displs(nprocs);
    MPI_Gather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_len = 0;
    if (rank == 0)
        for (int i = 0; i < nprocs; i++) { displs[i] = total_len; total_len += all_lens[i]; }

    vector<char> recv(total_len);
    MPI_Gatherv(local_buf.c_str(), local_len, MPI_CHAR,
                recv.data(), all_lens.data(), displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        string all(recv.begin(), recv.end());
        size_t pos = 0, found;
        while ((found = all.find('\n', pos)) != string::npos)
        {
            string g = all.substr(pos, found - pos);
            if (!g.empty()) { guesses.emplace_back(g); total_guesses++; }
            pos = found + 1;
        }
    }
}

// worker调用的入口（内部直接复用Generate的广播逻辑）
void PriorityQueue::GenerateWorker()
{
    // worker也需要走一遍Generate的广播流程
    // 构造一个空PT，让Generate的广播部分自然运行
    PT dummy;
    Generate(dummy);  // rank!=0时Generate只参与广播和Gather，不操作priority
}