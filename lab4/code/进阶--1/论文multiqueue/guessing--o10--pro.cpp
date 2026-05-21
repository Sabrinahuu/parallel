#include "PCFG.h"
#include <omp.h>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

//串行阈值
static const int PARALLEL_THRESHOLD = 4096;

// 最大堆比较器：prob 越大优先级越高
static auto cmp = [](const PT &a, const PT &b) {
    return a.prob < b.prob;
};

void PriorityQueue::DestroyMultiQueue()
{
    if (multiqueue_inited)
    {
        for (int i = 0; i < num_queues; i++)
        {
            omp_destroy_lock(&queue_locks[i]);
        }
        multiqueue_inited = false;
    }

    if (guess_lock_inited)
    {
        omp_destroy_lock(&guess_lock);
        guess_lock_inited = false;
    }
}

int PriorityQueue::RandomQueueId() const
{
    static thread_local mt19937 rng(
        (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count()
        + 1315423911u * (unsigned)(omp_get_thread_num() + 1)
    );

    uniform_int_distribution<int> dist(0, num_queues - 1);
    return dist(rng);
}

void PriorityQueue::RefreshTopProb(int qid)
{
    if (local_queues[qid].empty())
    {
        queue_top_probs[qid] = -1.0f;
    }
    else
    {
        queue_top_probs[qid] = local_queues[qid].front().prob;
    }
}


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
    DestroyMultiQueue();

    int p = omp_get_max_threads();
    num_queues = queue_factor * p;

    local_queues.clear();
    local_queues.resize(num_queues);

    queue_locks.clear();
    queue_locks.resize(num_queues);

    queue_top_probs.clear();
    queue_top_probs.resize(num_queues, -1.0f);

    for (int i = 0; i < num_queues; i++)
    {
        omp_init_lock(&queue_locks[i]);
    }

    omp_init_lock(&guess_lock);

    multiqueue_inited = true;
    guess_lock_inited = true;

    int qid = 0;

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();

        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(
                    m.letters[m.FindLetter(seg)].ordered_values.size()
                );
            }
            else if (seg.type == 2)
            {
                pt.max_indices.emplace_back(
                    m.digits[m.FindDigit(seg)].ordered_values.size()
                );
            }
            else if (seg.type == 3)
            {
                pt.max_indices.emplace_back(
                    m.symbols[m.FindSymbol(seg)].ordered_values.size()
                );
            }
        }

        pt.preterm_prob =
            float(m.preterm_freq[m.FindPT(pt)]) / float(m.total_preterm);

        CalProb(pt);

        // 轮转分配到多个局部队列
        local_queues[qid].emplace_back(std::move(pt));
        qid = (qid + 1) % num_queues;
    }

    // 每个局部队列分别建最大堆
    for (int i = 0; i < num_queues; i++)
    {
        make_heap(local_queues[i].begin(), local_queues[i].end(), cmp);
        RefreshTopProb(i);
    }
}

void PriorityQueue::PushPT(PT &&pt)
{
    while (true)
    {
        int qid = RandomQueueId();

        // 锁不上就换一个队列，避免固定等待
        if (omp_test_lock(&queue_locks[qid]))
        {
            local_queues[qid].emplace_back(std::move(pt));
            push_heap(local_queues[qid].begin(), local_queues[qid].end(), cmp);

            RefreshTopProb(qid);

            omp_unset_lock(&queue_locks[qid]);
            return;
        }
    }
}

bool PriorityQueue::TryPopBestOfTwo(PT &out)
{
    const int MAX_ATTEMPTS = 128;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
    {
        int i = RandomQueueId();
        int j = RandomQueueId();

        if (i == j)
            continue;

        float pi = queue_top_probs[i];
        float pj = queue_top_probs[j];

        int best = (pi >= pj) ? i : j;

        if (queue_top_probs[best] < 0.0f)
            continue;

        if (omp_test_lock(&queue_locks[best]))
        {
            if (!local_queues[best].empty())
            {
                pop_heap(local_queues[best].begin(), local_queues[best].end(), cmp);

                out = std::move(local_queues[best].back());
                local_queues[best].pop_back();

                RefreshTopProb(best);

                omp_unset_lock(&queue_locks[best]);
                return true;
            }

            RefreshTopProb(best);
            omp_unset_lock(&queue_locks[best]);
        }
    }

    // 兜底：随机失败后线性扫描，防止队列还有元素但抽不到
    for (int qid = 0; qid < num_queues; qid++)
    {
        if (omp_test_lock(&queue_locks[qid]))
        {
            if (!local_queues[qid].empty())
            {
                pop_heap(local_queues[qid].begin(), local_queues[qid].end(), cmp);

                out = std::move(local_queues[qid].back());
                local_queues[qid].pop_back();

                RefreshTopProb(qid);

                omp_unset_lock(&queue_locks[qid]);
                return true;
            }

            RefreshTopProb(qid);
            omp_unset_lock(&queue_locks[qid]);
        }
    }

    return false;
}

bool PriorityQueue::HasPT()
{
    for (int i = 0; i < num_queues; i++)
    {
        if (omp_test_lock(&queue_locks[i]))
        {
            bool not_empty = !local_queues[i].empty();
            omp_unset_lock(&queue_locks[i]);

            if (not_empty)
                return true;
        }
    }

    return false;
}

void PriorityQueue::PopNext()
{
    PT top_pt;

    // 从两个随机局部队列中取概率较大的 PT
    if (!TryPopBestOfTwo(top_pt))
    {
        return;
    }

    // Generate 会写共享 guesses 和 total_guesses，因此这里加锁保证正确性
    omp_set_lock(&guess_lock);
    Generate(top_pt);
    omp_unset_lock(&guess_lock);

    // 由当前 PT 扩展新 PT
    vector<PT> new_pts = top_pt.NewPTs();

    for (PT &pt : new_pts)
    {
        CalProb(pt);
        PushPT(std::move(pt));
    }
}



vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
        return res;

    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i += 1)
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
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];

        int n = pt.max_indices[0];

        //  小任务串行，避免进入 OpenMP 并行区域的固定开销
        if (n < PARALLEL_THRESHOLD)
        {
            guesses.reserve(guesses.size() + n);
            for (int i = 0; i < n; i++)
                guesses.emplace_back(a->ordered_values[i]);
            total_guesses += n;
            return;
        }

        // 预分配，保证并行写入时 vector 不会重新分配内存
        int base = (int)guesses.size();
        guesses.resize(base + n);

        // [改进3] 线程数不超过任务量，避免空转
        int t = min(omp_get_max_threads(), n);

        // [改进2] schedule(guided)：先分配大块，尾部分配小块，动态均衡负载
        //         对于单 segment PT，每次迭代只是一次 string 赋值（无拼接），
        //         工作量相对均匀，guided 和 static 差距不大，但 guided 更通用。
        #pragma omp parallel for schedule(guided) num_threads(t)
        for (int i = 0; i < n; i++)
            guesses[base + i] = a->ordered_values[i];

        total_guesses += n;
    }
    else
    {
        // 构造前缀（串行，segment 数量少，耗时可忽略）
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
            seg_idx += 1;
            if (seg_idx == (int)pt.content.size() - 1)
                break;
        }

        int last = (int)pt.content.size() - 1;
        segment *a = nullptr;
        if (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
        if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        if (pt.content[last].type == 3) a = &m.symbols[m.FindSymbol(pt.content[last])];

        int n = (int)pt.max_indices[last];

        // [改进1] 小任务串行
        if (n < PARALLEL_THRESHOLD)
        {
            guesses.reserve(guesses.size() + n);
            for (int i = 0; i < n; i++)
                guesses.emplace_back(guess + a->ordered_values[i]);
            total_guesses += n;
            return;
        }

        int base = (int)guesses.size();
        guesses.resize(base + n);


        int t = min(omp_get_max_threads(), n);


        #pragma omp parallel for schedule(guided) num_threads(t)
        for (int i = 0; i < n; i++)
            guesses[base + i] = guess + a->ordered_values[i];

        total_guesses += n;
    }
}