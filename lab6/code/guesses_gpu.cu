// guesses_gpu.cu  —— 批量口令猜测生成（GPU 并行）
#include "guesses_gpu.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

// ─────────────────────────────────────────────────────────────
// Kernel：每个线程处理一条猜测
//   guess_i = prefixes[task_id] + values[task_id][local_i]
//
// 数据布局（全部打包成扁平数组，避免 GPU 上用指针）：
//   d_prefix_flat  : 所有前缀连续存储
//   d_prefix_off   : 每个任务的前缀起始 offset
//   d_prefix_len   : 每个任务的前缀长度
//   d_value_flat   : 所有 value 字符串连续存储
//   d_value_off    : 第 g 条猜测的 value 起始 offset（全局编号）
//   d_value_len    : 第 g 条猜测的 value 长度
//   d_task_id      : 第 g 条猜测属于哪个任务（用于找前缀）
//   total_guesses  : 总猜测条数
//   d_out          : 输出，每条占 max_len 字节
// ─────────────────────────────────────────────────────────────
__global__ void BatchGenerateKernel(
    const char*          d_prefix_flat,
    const unsigned int*  d_prefix_off,
    const unsigned int*  d_prefix_len,
    const char*          d_value_flat,
    const unsigned int*  d_value_off,
    const unsigned int*  d_value_len,
    const int*           d_task_id,
    int                  total_guesses,
    char*                d_out,
    int                  max_len)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_guesses) return;

    char* out = d_out + (size_t)g * max_len;

    // 写前缀
    int tid    = d_task_id[g];
    int plen   = d_prefix_len[tid];
    const char* pfx = d_prefix_flat + d_prefix_off[tid];
    for (int k = 0; k < plen; ++k) out[k] = pfx[k];

    // 写 value
    int vlen   = d_value_len[g];
    const char* val = d_value_flat + d_value_off[g];
    for (int k = 0; k < vlen; ++k) out[plen + k] = val[k];

    out[plen + vlen] = '\0';
}

// ─────────────────────────────────────────────────────────────
// 主机端封装
// ─────────────────────────────────────────────────────────────
void GenerateBatchOnGPU(
    const std::vector<BatchTask>& tasks,
    std::vector<std::string>&     out_guesses,
    int&                          total_guesses)
{
    if (tasks.empty()) return;

    int ntasks = (int)tasks.size();

    // ── 1. 打包前缀 ──
    std::vector<unsigned int> h_prefix_off(ntasks);
    std::vector<unsigned int> h_prefix_len(ntasks);
    std::vector<char> h_prefix_flat;
    for (int t = 0; t < ntasks; ++t) {
        h_prefix_off[t] = (unsigned int)h_prefix_flat.size();
        h_prefix_len[t] = (unsigned int)tasks[t].prefix.size();
        for (char c : tasks[t].prefix) h_prefix_flat.push_back(c);
    }
    if (h_prefix_flat.empty()) h_prefix_flat.push_back('\0'); // 避免空分配

    // ── 2. 打包 value + 建立全局猜测索引 ──
    int total = 0;
    for (auto& t : tasks) total += (int)t.values.size();
    if (total == 0) return;

    std::vector<unsigned int> h_value_off(total);
    std::vector<unsigned int> h_value_len(total);
    std::vector<int>          h_task_id(total);
    std::vector<char>         h_value_flat;
    int max_len = 0;

    int g = 0;
    for (int t = 0; t < ntasks; ++t) {
        int plen = (int)tasks[t].prefix.size();
        for (auto& v : tasks[t].values) {
            h_value_off[g] = (unsigned int)h_value_flat.size();
            h_value_len[g] = (unsigned int)v.size();
            h_task_id[g]   = t;
            int guess_len  = plen + (int)v.size() + 1;
            if (guess_len > max_len) max_len = guess_len;
            for (char c : v) h_value_flat.push_back(c);
            ++g;
        }
    }
    if (h_value_flat.empty()) h_value_flat.push_back('\0');

    // ── 3. 分配 device 内存 ──
    char*         d_prefix_flat = nullptr;
    unsigned int* d_prefix_off  = nullptr;
    unsigned int* d_prefix_len  = nullptr;
    char*         d_value_flat  = nullptr;
    unsigned int* d_value_off   = nullptr;
    unsigned int* d_value_len   = nullptr;
    int*          d_task_id     = nullptr;
    char*         d_out         = nullptr;

    cudaMalloc(&d_prefix_flat, h_prefix_flat.size());
    cudaMalloc(&d_prefix_off,  ntasks * sizeof(unsigned int));
    cudaMalloc(&d_prefix_len,  ntasks * sizeof(unsigned int));
    cudaMalloc(&d_value_flat,  h_value_flat.size());
    cudaMalloc(&d_value_off,   total  * sizeof(unsigned int));
    cudaMalloc(&d_value_len,   total  * sizeof(unsigned int));
    cudaMalloc(&d_task_id,     total  * sizeof(int));
    cudaMalloc(&d_out,         (size_t)total * max_len);

    // ── 4. H→D ──
    cudaMemcpy(d_prefix_flat, h_prefix_flat.data(), h_prefix_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_off,  h_prefix_off.data(),  ntasks * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_len,  h_prefix_len.data(),  ntasks * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_flat,  h_value_flat.data(),  h_value_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_off,   h_value_off.data(),   total  * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_len,   h_value_len.data(),   total  * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_task_id,     h_task_id.data(),     total  * sizeof(int), cudaMemcpyHostToDevice);

    // ── 5. Kernel ──
    int block = 256;
    int grid  = (total + block - 1) / block;
    BatchGenerateKernel<<<grid, block>>>(
        d_prefix_flat, d_prefix_off, d_prefix_len,
        d_value_flat,  d_value_off,  d_value_len,
        d_task_id, total, d_out, max_len);
    cudaDeviceSynchronize();

    // ── 6. D→H ──
    std::vector<char> h_out((size_t)total * max_len);
    cudaMemcpy(h_out.data(), d_out, (size_t)total * max_len, cudaMemcpyDeviceToHost);

    // ── 7. 转回 vector<string> ──
    out_guesses.reserve(out_guesses.size() + total);
    for (int i = 0; i < total; ++i) {
        out_guesses.emplace_back(h_out.data() + (size_t)i * max_len);
    }
    total_guesses += total;

    // ── 8. 释放 ──
    cudaFree(d_prefix_flat); cudaFree(d_prefix_off); cudaFree(d_prefix_len);
    cudaFree(d_value_flat);  cudaFree(d_value_off);  cudaFree(d_value_len);
    cudaFree(d_task_id);     cudaFree(d_out);
}
