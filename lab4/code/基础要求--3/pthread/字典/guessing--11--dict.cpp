
#include "PCFG.h"
#include "str_dict.h"
#include <pthread.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

// 全局字典实例，init() 末尾构建，Generate() 全程使用
static StrDict dict;


//  FastFindSeg：带缓存的 segment 查找，首次 O(n) 之后 O(1)
static inline int SegKey(int type, int length) {
    return (type << 16) | length;
}

static const segment* FastFindSeg(const model &m, int type, int length)
{
    static unordered_map<int, const segment*> cache;
    int key = SegKey(type, length);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const segment *result = nullptr;
    if (type == 1) {
        for (const auto &s : m.letters)
            if (s.length == length) { result = &s; break; }
    } else if (type == 2) {
        for (const auto &s : m.digits)
            if (s.length == length) { result = &s; break; }
    } else if (type == 3) {
        for (const auto &s : m.symbols)
            if (s.length == length) { result = &s; break; }
    }
    cache[key] = result;
    return result;
}


// pthread 并行部分
// GuessWorker 改为从字典连续内存（dict_base）读取字符串
namespace {

struct GuessTaskArg {
    const string  *prefix;
    const char    *dict_base;   // 字典基址：指向该 segment 第0个字符串
    int            str_len;     // 每个字符串的字节长度
    vector<string>*out;
    size_t         out_offset;
    size_t         begin;
    size_t         end;
};

// 直接从连续内存读，减少 vector<string> 的间接寻址开销
void *GuessWorker(void *ptr)
{
    GuessTaskArg  *arg    = static_cast<GuessTaskArg*>(ptr);
    const string  &prefix = *(arg->prefix);
    vector<string>&out    = *(arg->out);
    const char    *base   = arg->dict_base;
    int            slen   = arg->str_len;

    if (prefix.empty()) {
        for (size_t i = arg->begin; i < arg->end; ++i)
            out[arg->out_offset + i].assign(base + i * slen, slen);
    } else {
        string tmp;
        tmp.reserve(prefix.size() + slen);
        for (size_t i = arg->begin; i < arg->end; ++i) {
            tmp = prefix;
            tmp.append(base + i * slen, slen);
            out[arg->out_offset + i] = tmp;
        }
    }
    return nullptr;
}

size_t PickThreadNum(size_t n)
{
    long cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu <= 0) cpu = 4;
    return max<size_t>(1, min((size_t)cpu, n));
}

// [改动4] 新增 dict_base / str_len 参数，大任务走字典连续内存路径
void ParallelAppendGuesses(const string  &prefix,
                           const char    *dict_base,
                           int            str_len,
                           size_t         n,
                           vector<string>&out)
{
    if (n == 0) return;

    const size_t kMinParallel = 4096;
    if (n < kMinParallel) {
        // 小任务串行，直接用字典指针
        out.reserve(out.size() + n);
        if (prefix.empty()) {
            for (size_t i = 0; i < n; ++i)
                out.emplace_back(dict_base + i * str_len, str_len);
        } else {
            string tmp;
            tmp.reserve(prefix.size() + str_len);
            for (size_t i = 0; i < n; ++i) {
                tmp = prefix;
                tmp.append(dict_base + i * str_len, str_len);
                out.emplace_back(tmp);
            }
        }
        return;
    }

    const size_t old_size   = out.size();
    out.resize(old_size + n);

    const size_t thread_num = PickThreadNum(n);
    vector<pthread_t>    tids(thread_num);
    vector<GuessTaskArg> args(thread_num);

    const size_t chunk = (n + thread_num - 1) / thread_num;
    size_t real_threads = 0;

    for (size_t t = 0; t < thread_num; ++t) {
        const size_t begin = t * chunk;
        const size_t end   = min(n, begin + chunk);
        if (begin >= end) break;

        args[t] = GuessTaskArg{&prefix, dict_base, str_len, &out, old_size, begin, end};
        int rc = pthread_create(&tids[t], nullptr, GuessWorker, &args[t]);
        if (rc != 0) {
            // 创建失败，回退串行
            for (size_t i = begin; i < n; ++i) {
                if (prefix.empty())
                    out[old_size + i].assign(dict_base + i * str_len, str_len);
                else {
                    out[old_size + i] = prefix;
                    out[old_size + i].append(dict_base + i * str_len, str_len);
                }
            }
            break;
        }
        ++real_threads;
    }

    for (size_t t = 0; t < real_threads; ++t)
        pthread_join(tids[t], nullptr);
}

} // namespace


//  CalProb：FastFindSeg 替换 FindLetter/FindDigit/FindSymbol
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        int type   = pt.content[index].type;
        int length = pt.content[index].length;
        const segment *seg = FastFindSeg(m, type, length);
        if (seg) {
            pt.prob *= seg->ordered_freqs[idx];
            pt.prob /= seg->total_freq;
        }
        index++;
    }
}


// 末尾调用 dict.Build(m)
void PriorityQueue::init()
{
    // 先构建字典：只构建字符串连续内存，不参与概率计算
    dict.Build(m);
    cout << "StrDict built, data size: " << dict.dict_data.size() << " bytes" << endl;

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
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i])
        { pivot = i; res.emplace_back(*this); }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}


//  Generate：用字典查找替换 FindLetter/ordered_values 访问
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        int type   = pt.content[0].type;
        int length = pt.content[0].length;
        int n      = pt.max_indices[0];

        //取字典基址，rank=0 即概率最高的字符串起始位置
        const char *base = dict.Lookup(type, length, 0);
        if (!base) return;

        ParallelAppendGuesses("", base, length, (size_t)n, guesses);
        total_guesses += n;
    }
    else
    {
        // 构造前缀：dict.LookupString O(1) 查表代替 FindLetter + ordered_values
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            int type   = pt.content[seg_idx].type;
            int length = pt.content[seg_idx].length;
            guess += dict.LookupString(type, length, idx);
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        int last   = (int)pt.content.size() - 1;
        int type   = pt.content[last].type;
        int length = pt.content[last].length;
        int n      = pt.max_indices[last];

        const char *base = dict.Lookup(type, length, 0);
        if (!base) return;

        ParallelAppendGuesses(guess, base, length, (size_t)n, guesses);
        total_guesses += n;
    }
}