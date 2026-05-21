#include "PCFG.h"
#include <pthread.h>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

/*
 * pthread 并行生成猜测用的参数结构体
 */
struct GenerateArgs
{
    const segment *seg;
    const string *prefix;

    vector<string> *guesses;     // 直接指向总 guesses
    int *total_guesses;          // 总计数

    pthread_mutex_t *mutex;      // 保护 guesses 和 total_guesses

    int start;
    int end;
};

/*
 * 每个 pthread 线程真正执行的函数
 * 负责生成 [start, end) 范围内的猜测
 * 然后写入local[i]
 */
void *GenerateWorker(void *arg)
{
    GenerateArgs *args = (GenerateArgs *)arg;

    // 每个线程先生成自己的局部结果
    vector<string> local;
    local.reserve(args->end - args->start);

    for (int i = args->start; i < args->end; i++)
    {
        local.emplace_back(*(args->prefix) + args->seg->ordered_values[i]);
    }

    // 只在合并时加锁，避免每生成一个字符串都加锁
    pthread_mutex_lock(args->mutex);

    args->guesses->insert(
        args->guesses->end(),
        local.begin(),
        local.end()
    );

    *(args->total_guesses) += local.size();

    pthread_mutex_unlock(args->mutex);

    return nullptr;
}

/*
 * 通用并行生成函数
 * prefix + seg->ordered_values[i] 生成完整猜测
 */
static void GenerateParallel(
    const string &prefix,
    const segment *a,
    int n,
    vector<string> &guesses,
    int &total_guesses
)
{
    if (a == nullptr || n <= 0)
    {
        return;
    }

    const int THREAD_NUM = 4;
    int thread_count = min(THREAD_NUM, n);

    vector<pthread_t> threads(thread_count);
    vector<GenerateArgs> args(thread_count);

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, nullptr);

    for (int t = 0; t < thread_count; t++)
    {
        int start = t * n / thread_count;
        int end = (t + 1) * n / thread_count;

        args[t].seg = a;
        args[t].prefix = &prefix;
        args[t].guesses = &guesses;
        args[t].total_guesses = &total_guesses;
        args[t].mutex = &mutex;
        args[t].start = start;
        args[t].end = end;

        pthread_create(&threads[t], nullptr, GenerateWorker, &args[t]);
    }

    for (int t = 0; t < thread_count; t++)
    {
        pthread_join(threads[t], nullptr);
    }

    pthread_mutex_destroy(&mutex);
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
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }

            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }

            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
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

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;

        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }

        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }

        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        string prefix = "";
        GenerateParallel(prefix, a, pt.max_indices[0], guesses, total_guesses);
    }
    else
    {
        string guess;
        int seg_idx = 0;

        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }

            seg_idx += 1;

            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a = nullptr;
        int last = pt.content.size() - 1;

        if (pt.content[last].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last])];
        }

        if (pt.content[last].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last])];
        }

        if (pt.content[last].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last])];
        }

        GenerateParallel(
            guess,
            a,
            pt.max_indices[last],
            guesses,
            total_guesses
        );
    }
}