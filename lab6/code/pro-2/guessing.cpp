#include "guesses_gpu.h"
#include "PCFG.h"
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

// ═══════════════════════════════════════════════════════════════
// 双缓冲 + 后台 GPU 线程
// 缓冲区 0 和 1 交替使用：
//   CPU 向 g_batch[g_active] 写入任务
//   后台线程从 g_batch[1-g_active] 读取并发给 GPU
// ═══════════════════════════════════════════════════════════════

static const int BATCH_THRESHOLD = 200000; // 每批猜测数量阈值，可调大

// 双缓冲
static vector<BatchTask> g_batch[2];
static int  g_batch_count[2] = {0, 0};
static int  g_active = 0; // CPU 当前写入的缓冲区下标

// GPU 输出缓存（后台线程写入，主线程定期合并）
static vector<string> g_gpu_out;
static int            g_gpu_total = 0;
static mutex          g_out_mtx;  // 保护 g_gpu_out / g_gpu_total

// 后台线程控制
static thread             g_gpu_thread;
static mutex              g_task_mtx;
static condition_variable g_task_cv;
static bool g_task_ready = false; // 有新任务待处理
static bool g_shutdown   = false; // 通知后台线程退出
static int  g_pending    = 0;     // 待处理的缓冲区下标

// ── 后台 GPU 线程主函数 ──
static void GPUWorker()
{
    while (true)
    {
        int buf_idx = -1;
        {
            unique_lock<mutex> lk(g_task_mtx);
            g_task_cv.wait(lk, []{ return g_task_ready || g_shutdown; });
            if (g_shutdown && !g_task_ready) break;
            buf_idx      = g_pending;
            g_task_ready = false;
        }

        // 在后台调用 GPU
        vector<string> local_out;
        int local_total = 0;
        GenerateBatchOnGPU(g_batch[buf_idx], local_out, local_total);

        // 清空已处理的缓冲区
        g_batch[buf_idx].clear();
        g_batch_count[buf_idx] = 0;

        // 将结果合并到输出缓存
        {
            lock_guard<mutex> lk(g_out_mtx);
            for (auto& s : local_out)
                g_gpu_out.emplace_back(move(s));
            g_gpu_total += local_total;
        }
    }
}

// ── 启动后台线程（在 init() 之后、主循环之前调用） ──
void PriorityQueue::StartGPUWorker()
{
    g_shutdown   = false;
    g_task_ready = false;
    g_active     = 0;
    g_batch[0].clear(); g_batch[1].clear();
    g_batch_count[0] = g_batch_count[1] = 0;
    g_gpu_out.clear();
    g_gpu_total = 0;
    g_gpu_thread = thread(GPUWorker);
}

// ── 停止后台线程，处理剩余任务 ──
void PriorityQueue::StopGPUWorker()
{
    // flush 当前活跃缓冲区里的剩余任务
    FlushRemaining();

    // 通知后台线程退出
    {
        lock_guard<mutex> lk(g_task_mtx);
        g_shutdown = true;
    }
    g_task_cv.notify_one();
    if (g_gpu_thread.joinable()) g_gpu_thread.join();

    // 合并最终输出
    MergeGPUOutput();
}

// ── 将 GPU 输出合并到 PriorityQueue::guesses ──
void PriorityQueue::MergeGPUOutput()
{
    lock_guard<mutex> lk(g_out_mtx);
    for (auto& s : g_gpu_out)
        guesses.emplace_back(move(s));
    total_guesses += g_gpu_total;
    g_gpu_out.clear();
    g_gpu_total = 0;
}

// ── 把当前活跃缓冲区提交给后台线程，切换到另一个缓冲区 ──
static void SubmitBatch()
{
    int submit = g_active;
    g_active   = 1 - g_active; // 切换到另一个缓冲区，CPU 继续写

    {
        lock_guard<mutex> lk(g_task_mtx);
        g_pending    = submit;
        g_task_ready = true;
    }
    g_task_cv.notify_one();
}

// ── 处理剩余不足阈值的任务（主循环结束时调用） ──
void PriorityQueue::FlushRemaining()
{
    if (g_batch_count[g_active] > 0)
        SubmitBatch();
}

// ═══════════════════════════════════════════════════════════════
// PriorityQueue 原有函数
// ═══════════════════════════════════════════════════════════════

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
    if (content.size() == 1) return res;

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

// Generate：把任务加入当前活跃缓冲区，攒够后提交给后台 GPU 线程
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    BatchTask task;

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        task.prefix = "";
        task.values = a->ordered_values;
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
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        segment *a;
        if (pt.content[pt.content.size()-1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size()-1])];
        if (pt.content[pt.content.size()-1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size()-1])];
        if (pt.content[pt.content.size()-1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size()-1])];

        task.prefix = guess;
        task.values = a->ordered_values;
    }

    g_batch_count[g_active] += (int)task.values.size();
    g_batch[g_active].emplace_back(move(task));

    // 攒够阈值就提交给后台线程，CPU 立刻切换到另一个缓冲区继续工作
    if (g_batch_count[g_active] >= BATCH_THRESHOLD)
        SubmitBatch();
}
