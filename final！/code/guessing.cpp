#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <mpi.h>
using namespace std;
using namespace chrono;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            int seg_id = m.FindLetter(pt.content[index]);
            pt.prob *= m.letters[seg_id].ordered_freqs[idx];
            pt.prob /= m.letters[seg_id].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            int seg_id = m.FindDigit(pt.content[index]);
            pt.prob *= m.digits[seg_id].ordered_freqs[idx];
            pt.prob /= m.digits[seg_id].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            int seg_id = m.FindSymbol(pt.content[index]);
            pt.prob *= m.symbols[seg_id].ordered_freqs[idx];
            pt.prob /= m.symbols[seg_id].total_freq;
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

void PriorityQueue::Generate(PT pt)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    CalProb(pt);

    string guess = "";
    segment *a = nullptr;
    int total = 0;

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

    int prefix_len = guess.size();
    int meta[4] = {prefix_len, total, a->type, a->length};
    MPI_Bcast(meta, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (prefix_len > 0)
    {
        MPI_Bcast(&guess[0], prefix_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    if (rank != 0)
    {
        int seg_type = meta[2];
        int seg_len = meta[3];
        segment tmp(seg_type, seg_len);
        if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
        if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
        if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];
    }

    int chunk = total / nprocs;
    int start = rank * chunk;
    int end = (rank == nprocs - 1) ? total : start + chunk;

    vector<string> local_guesses;
    local_guesses.reserve(end - start);
    for (int i = start; i < end; i++)
    {
        local_guesses.emplace_back();
        string &candidate = local_guesses.back();
        candidate.reserve(guess.size() + a->ordered_values[i].size());
        candidate.append(guess);
        candidate.append(a->ordered_values[i]);
    }

    auto start_hash = system_clock::now();
    if (!local_guesses.empty())
        MD5HashBatch_NEON(local_guesses, nullptr);
    auto end_hash = system_clock::now();
    auto duration_hash = duration_cast<microseconds>(end_hash - start_hash);
    double local_hash_time = double(duration_hash.count()) * microseconds::period::num / microseconds::period::den;

    int local_count = local_guesses.size();
    int global_count = 0;
    double global_hash_time = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_time, &global_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        total_guesses += global_count;
        hash_time += global_hash_time;
    }
}

void PriorityQueue::GenerateWorker()
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int meta[4];
    MPI_Bcast(meta, 4, MPI_INT, 0, MPI_COMM_WORLD);

    int prefix_len = meta[0];
    int total = meta[1];
    int seg_type = meta[2];
    int seg_len = meta[3];

    string guess = "";
    if (prefix_len > 0)
    {
        guess.resize(prefix_len);
        MPI_Bcast(&guess[0], prefix_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    segment tmp(seg_type, seg_len);
    segment *a = nullptr;
    if (seg_type == 1) a = &m.letters[m.FindLetter(tmp)];
    if (seg_type == 2) a = &m.digits[m.FindDigit(tmp)];
    if (seg_type == 3) a = &m.symbols[m.FindSymbol(tmp)];

    int chunk = total / nprocs;
    int start = rank * chunk;
    int end = (rank == nprocs - 1) ? total : start + chunk;

    vector<string> local_guesses;
    local_guesses.reserve(end - start);
    for (int i = start; i < end; i++)
    {
        local_guesses.emplace_back();
        string &candidate = local_guesses.back();
        candidate.reserve(guess.size() + a->ordered_values[i].size());
        candidate.append(guess);
        candidate.append(a->ordered_values[i]);
    }

    auto start_hash = system_clock::now();
    if (!local_guesses.empty())
        MD5HashBatch_NEON(local_guesses, nullptr);
    auto end_hash = system_clock::now();
    auto duration_hash = duration_cast<microseconds>(end_hash - start_hash);
    double local_hash_time = double(duration_hash.count()) * microseconds::period::num / microseconds::period::den;

    int local_count = local_guesses.size();
    int global_count = 0;
    double global_hash_time = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_time, &global_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}
