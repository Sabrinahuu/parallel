#include "PCFG.h"
#include <pthread.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

namespace {
struct GuessTaskArg {
    const string *prefix;//当前PT已经确定好的前缀
    const vector<string> *values;//最后一个segment的所有候选value
    vector<string> *out;//输出目标
    size_t out_offset;//当前这一轮 Generate 写入 guesses 的起始位置
    size_t begin;//当前线程起始下标
    size_t end;//当前线程结束下标
};

//每个线程真正执行的函数
void *GuessWorker(void *ptr)
{
    GuessTaskArg *arg = static_cast<GuessTaskArg *>(ptr);
    //取出指针，变成更方便使用的引用
    const string &prefix = *(arg->prefix);
    const vector<string> &values = *(arg->values);
    vector<string> &out = *(arg->out);
    
      /*
     * 如果 prefix 为空，说明当前 PT 只有一个 segment
     *不需要拼接前缀，直接写入
     */
    if (prefix.empty()) {
        for (size_t i = arg->begin; i < arg->end; ++i) {
            out[arg->out_offset + i] = values[i];
        }
    } 
    /*
     * 如果 prefix 不为空，说明当前 PT 有多个 segment
     * 前面的 segment 已经被实例化为 prefix
     * 当前线程只需要枚举最后一个 segment 并拼接
     */
    else {
        for (size_t i = arg->begin; i < arg->end; ++i) {
            out[arg->out_offset + i] = prefix + values[i];
        }
    }
    return nullptr;
}

/*
 * 根据任务数量 n 和 CPU 核心数决定线程数
 *
 * 线程数不应该无限开：
 * 1. 超过 CPU 核心数通常收益不大；
 * 2. 如果 n 很小，线程数也不应超过任务数；
 * 3. pthread 创建和回收本身有开销
 */
size_t PickThreadNum(size_t n)
{   //读取当前系统中在线可用的CPU核心数
    long cpu = sysconf(_SC_NPROCESSORS_ONLN);
    //如果获取失败，保守地假设有 4 个核心
    if (cpu <= 0) cpu = 4;
    size_t t = static_cast<size_t>(cpu);
    //线程数最多为 CPU 核心数，同时不能超过任务数量 n
    return max<size_t>(1, min(t, n));
}

/*
 *真正封装 pthread 并行生成猜测的函数。
 *
 * 参数含义：
 * prefix: 已经确定的前缀;如果 PT 只有一个 segment，则 prefix 为空
 * values: 最后一个 segment 的候选 value 列表
 * n: 本轮需要生成的候选数量
 * out: 输出到 PriorityQueue::guesses
 */
void ParallelAppendGuesses(const string &prefix,
                           const vector<string> &values,
                           size_t n,
                           vector<string> &out)
{
    if (n == 0) return;

    // 小任务不启线程，避免 pthread 创建/回收成本超过收益。
    const size_t kMinParallel = 4096;//经验阈值
    if (n < kMinParallel) {
        out.reserve(out.size() + n);
        if (prefix.empty()) {
            for (size_t i = 0; i < n; ++i) out.emplace_back(values[i]);
        } else {
            for (size_t i = 0; i < n; ++i) out.emplace_back(prefix + values[i]);
        }
        return;
    }
    
    //记录本轮Generate开始写入的位置
    const size_t old_size = out.size();
    //先resize分配好n个位置，使得每个线程直接通过下标写入
    out.resize(old_size + n);
    
    //根据 CPU 核心数和任务量选择线程数
    const size_t thread_num = PickThreadNum(n);
    vector<pthread_t> tids(thread_num);
    //每个线程的参数
    vector<GuessTaskArg> args(thread_num);
    
    //按照负载均衡策略静态均匀分块：在一个 PT 内部按最后一个 segment 的 value 下标均匀切分
    const size_t chunk = (n + thread_num - 1) / thread_num;
    size_t real_threads = 0;
    for (size_t t = 0; t < thread_num; ++t) {
        const size_t begin = t * chunk;
        const size_t end = min(n, begin + chunk);
        if (begin >= end) break;
        
        //给第t个线程准备参数
        //每个线程负责 values[begin, end)
        //写入位置从 out_offset + begin 开始
        args[t] = GuessTaskArg{
            &prefix, 
            &values, 
            &out, 
            old_size, 
            begin, 
            end
        };
        //创建线程
        int rc = pthread_create(&tids[t], nullptr, GuessWorker, &args[t]);
        if (rc != 0) {
            // 创建失败时，当前分片回退为串行，后续分片也串行处理。
            if (prefix.empty()) {
                for (size_t i = begin; i < n; ++i) out[old_size + i] = values[i];
            } else {
                for (size_t i = begin; i < n; ++i) out[old_size + i] = prefix + values[i];
            }
            break;
        }
        ++real_threads;
    }
    
    //等到所有成功创建的线程结束
    for (size_t t = 0; t < real_threads; ++t) {
        pthread_join(tids[t], nullptr);
    }
}
} // namespace

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        //字母
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        //数字
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        //符号
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

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

        priority.emplace_back(pt);
    }

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

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// pthread 版本：只并行化“最后一个 segment 的所有 value 拼接”这一步。
// 优点：不并行修改 priority queue，不破坏 PCFG 出队顺序；线程只写 guesses 的不同下标，无需锁
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
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

        size_t n = static_cast<size_t>(pt.max_indices[0]);
        ParallelAppendGuesses("", a->ordered_values, n, guesses);
        total_guesses += static_cast<int>(n);
    }
    else
    {
        string guess;
        int seg_idx = 0;

        // 给当前PT的所有segment赋予实际的值（最后一个segment除外）
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

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        size_t last = pt.content.size() - 1;
        segment *a = nullptr;
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

        size_t n = static_cast<size_t>(pt.max_indices[last]);
        ParallelAppendGuesses(guess, a->ordered_values, n, guesses);
        total_guesses += static_cast<int>(n);
    }
}
