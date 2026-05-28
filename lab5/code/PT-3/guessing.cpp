#include "PCFG.h"
#include <mpi.h>
#include <cstring>
using namespace std;

#define BATCH_SIZE 64

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
                { priority.emplace(iter + 1, pt); break; }
            }
            if (iter == priority.end() - 1)
            { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob)
            { priority.emplace(iter, pt); break; }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++)
    {
        curr_indices[i] += 1;
        if (curr_indices[i] < max_indices[i])
        { pivot = i; res.emplace_back(*this); }
        curr_indices[i] -= 1;
    }
    pivot = init_pivot;
    return res;
}

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
        { guesses.emplace_back(a->ordered_values[i]); total_guesses++; }
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
        { guesses.emplace_back(guess + a->ordered_values[i]); total_guesses++; }
    }
}

// PT信息打包结构体，用于一次性广播
struct PTInfo {
    char prefix[256];
    int  prefix_len;
    int  seg_type;
    int  seg_len;
    int  total;
};

// ============================================================
// PopNextBatch：核心并行函数
//
// 每轮只有2次MPI通信：
//   1. MPI_Bcast 广播batch_size和PTInfo数组
//   2. MPI_Allreduce 同步总猜测数
//
// 各进程直接写入自己的guesses，完全消除Gatherv开销
// 进程r负责编号为 r, r+nprocs, r+2*nprocs ... 的PT（轮询分配）
// ============================================================
void PriorityQueue::PopNextBatch()
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Step 1: 进程0构建PTInfo数组
    int batch_size = 0;
    vector<PTInfo> pt_infos;

    if (rank == 0)
    {
        batch_size = min(BATCH_SIZE, (int)priority.size());
        pt_infos.resize(batch_size);

        for (int i = 0; i < batch_size; i++)
        {
            PT &pt = priority[i];
            CalProb(pt);
            PTInfo &info = pt_infos[i];
            memset(&info, 0, sizeof(PTInfo));

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
            info.prefix_len = min((int)guess.size(), 255);
            memcpy(info.prefix, guess.c_str(), info.prefix_len);

            int last = pt.content.size() - 1;
            info.seg_type = pt.content[last].type;
            info.seg_len  = pt.content[last].length;
            info.total    = pt.max_indices[last];
        }
    }

    // Step 2: 一次性广播所有PT信息（只有2次MPI调用）
    MPI_Bcast(&batch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (batch_size == 0) return;

    if (rank != 0) pt_infos.resize(batch_size);
    MPI_Bcast(pt_infos.data(), batch_size * sizeof(PTInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Step 3: 各进程轮询处理自己负责的PT，直接写入guesses（无通信）
    for (int i = rank; i < batch_size; i += nprocs)
    {
        PTInfo &info = pt_infos[i];
        string guess(info.prefix, info.prefix_len);

        segment tmp(info.seg_type, info.seg_len);
        segment *a = nullptr;
        if (info.seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
        if (info.seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
        if (info.seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];
        if (a == nullptr) continue;

        for (int j = 0; j < info.total; j++)
            guesses.emplace_back(guess + a->ordered_values[j]);
    }

    // Step 4: 进程0更新优先队列（无通信）
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
                if (iter != priority.end() - 1 &&
                    pt.prob <= iter->prob && pt.prob > (iter+1)->prob)
                { priority.emplace(iter+1, pt); inserted = true; break; }
            }
            if (!inserted) priority.emplace_back(pt);
        }
    }
}