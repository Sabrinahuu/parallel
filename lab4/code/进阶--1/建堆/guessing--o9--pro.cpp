#include "PCFG.h"
#include <omp.h>
#include <algorithm>
using namespace std;

// 串行阈值
static const int PARALLEL_THRESHOLD = 4096;

// 堆的比较器：概率小的排后面（最大堆，堆顶是概率最大的PT）
static auto cmp = [](const PT &a, const PT &b) {
    return a.prob < b.prob;
};

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

// 构建一个按概率排序的优先队列
void PriorityQueue::init()
{
    priority.clear();
    priority.reserve(m.ordered_pts.size());

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();

        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            else if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            else if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / float(m.total_preterm);
        CalProb(pt);
        priority.emplace_back(pt);
    }

    // 用 make_heap 一次性建堆，O(n)，比逐个 push_heap 的 O(n log n) 更快
    make_heap(priority.begin(), priority.end(), cmp);
}

void PriorityQueue::PopNext()
{
    // 1. 将堆顶（概率最大的PT）移到 vector 末尾，堆缩小1，O(log n)
    pop_heap(priority.begin(), priority.end(), cmp);
    PT top_pt = move(priority.back());
    priority.pop_back();

    // 2. 用该PT生成猜测
    Generate(top_pt);

    // 3. 生成新PT并逐个以 O(log n) 插入堆
    //    原版线性扫描插入为 O(n)，改为 push_heap 后为 O(log n)
    vector<PT> new_pts = top_pt.NewPTs();
    for (PT &pt : new_pts)
    {
        CalProb(pt);
        priority.emplace_back(move(pt));
        // push_heap：将新加入末尾的元素上浮到正确位置，O(log n)
        push_heap(priority.begin(), priority.end(), cmp);
    }
}

// NewPTs：与原版完全相同
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

        if (n < PARALLEL_THRESHOLD)
        {
            guesses.reserve(guesses.size() + n);
            for (int i = 0; i < n; i++)
                guesses.emplace_back(a->ordered_values[i]);
            total_guesses += n;
            return;
        }

        int base = (int)guesses.size();
        guesses.resize(base + n);
        int t = min(omp_get_max_threads(), n);

        #pragma omp parallel for schedule(guided) num_threads(t)
        for (int i = 0; i < n; i++)
            guesses[base + i] = a->ordered_values[i];

        total_guesses += n;
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