
#include "PCFG.h"
#include <omp.h>
#include <algorithm>
using namespace std;

//串行阈值
static const int PARALLEL_THRESHOLD = 4096;

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

//构建一个按概率排序的优先队列
void PriorityQueue::init()
{   
    //清空优先队列并预分配内存
    priority.clear();
    priority.reserve(m.ordered_pts.size());
    
    //遍历PT
    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();
        
        //为每个片段记录其候选值的数量
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
        
        //计算概率
        pt.preterm_prob =
            float(m.preterm_freq[m.FindPT(pt)]) / float(m.total_preterm);

        CalProb(pt);

        priority.emplace_back(pt);
    }

    //按概率降序排序
    sort(priority.begin(), priority.end(), [](const PT &a, const PT &b) {
        return a.prob > b.prob;
    });
}


void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
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

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// -----------------------------------------------------------------------
// NewPTs：与原版完全相同，不做修改
// -----------------------------------------------------------------------
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

        //小任务串行
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

        //线程数适配任务量
        int t = min(omp_get_max_threads(), n);


        #pragma omp parallel for schedule(guided) num_threads(t)
        for (int i = 0; i < n; i++)
            guesses[base + i] = guess + a->ordered_values[i];

        total_guesses += n;
    }
}