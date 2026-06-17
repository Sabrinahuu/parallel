// guesses_gpu.cu
// GPU并行化口令猜测生成：针对 PriorityQueue::Generate() 中的两个 for 循环

#include "guesses_gpu.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

// ─────────────────────────────────────────────
// CUDA kernel
//
// 每个线程负责一个 value（对应原始循环中的一次迭代），
// 将 prefix + values[i] 拼接写入输出缓冲区。
//
// 参数说明：
//   d_values_flat : 所有 value 字符串连续存储（无 '\0' 间隔，用 offset 定位）
//   d_offsets     : d_offsets[i] = 第 i 个 value 在 d_values_flat 中的起始字节
//   d_lengths     : d_lengths[i] = 第 i 个 value 的字节长度
//   n             : value 总数
//   d_prefix      : 前缀字符串（已拼接好除最后一个 segment 以外的部分）
//   prefix_len    : 前缀长度
//   d_out_flat    : 输出缓冲区（每条猜测占 MAX_GUESS_LEN 字节，'\0' 结尾）
// ─────────────────────────────────────────────
__global__ void GenerateKernel(
    const char*          d_values_flat,
    const unsigned int*  d_offsets,
    const unsigned int*  d_lengths,
    int                  n,
    const char*          d_prefix,
    int                  prefix_len,
    char*                d_out_flat,
    int                  max_guess_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    char* out = d_out_flat + (size_t)i * max_guess_len;

    // 写前缀
    for (int k = 0; k < prefix_len; ++k)
        out[k] = d_prefix[k];

    // 写 value
    const char* val = d_values_flat + d_offsets[i];
    unsigned int vlen = d_lengths[i];
    for (unsigned int k = 0; k < vlen; ++k)
        out[prefix_len + k] = val[k];

    // 写终止符
    out[prefix_len + vlen] = '\0';
}

// ─────────────────────────────────────────────
// 主机端封装函数
// ─────────────────────────────────────────────
void GenerateOnGPU(
    const std::vector<std::string>& values,   // 最后一个 segment 的所有 value
    const std::string&              prefix,   // 已拼好的前缀（单 segment 时为空）
    std::vector<std::string>&       out_guesses,
    int&                            total_guesses)
{
    int n = (int)values.size();
    if (n == 0) return;

    // ── 1. 计算各 value 的 offset / length，打包成扁平字符数组 ──
    std::vector<unsigned int> h_offsets(n);
    std::vector<unsigned int> h_lengths(n);
    unsigned int total_chars = 0;
    for (int i = 0; i < n; ++i) {
        h_offsets[i] = total_chars;
        h_lengths[i] = (unsigned int)values[i].size();
        total_chars  += h_lengths[i];
    }

    std::vector<char> h_values_flat(total_chars);
    for (int i = 0; i < n; ++i)
        memcpy(h_values_flat.data() + h_offsets[i],
               values[i].data(), h_lengths[i]);

    // ── 2. 计算输出缓冲区所需的单条最大长度 ──
    int prefix_len   = (int)prefix.size();
    int max_val_len  = 0;
    for (int i = 0; i < n; ++i)
        if ((int)h_lengths[i] > max_val_len) max_val_len = h_lengths[i];
    int max_guess_len = prefix_len + max_val_len + 1; // +1 for '\0'

    // ── 3. 分配 device 内存 ──
    char*         d_values_flat = nullptr;
    unsigned int* d_offsets     = nullptr;
    unsigned int* d_lengths     = nullptr;
    char*         d_prefix      = nullptr;
    char*         d_out_flat    = nullptr;

    cudaMalloc(&d_values_flat, total_chars);
    cudaMalloc(&d_offsets,     n * sizeof(unsigned int));
    cudaMalloc(&d_lengths,     n * sizeof(unsigned int));
    cudaMalloc(&d_prefix,      prefix_len + 1);
    cudaMalloc(&d_out_flat,    (size_t)n * max_guess_len);

    // ── 4. H→D 拷贝 ──
    cudaMemcpy(d_values_flat, h_values_flat.data(), total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,     h_offsets.data(),     n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,     h_lengths.data(),     n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix,      prefix.data(),        prefix_len, cudaMemcpyHostToDevice);
    // 将前缀末尾的 '\0' 也拷贝（prefix_len 可为 0，无害）
    char zero = '\0';
    cudaMemcpy(d_prefix + prefix_len, &zero, 1, cudaMemcpyHostToDevice);

    // ── 5. 启动 kernel ──
    int block_size = 256;
    int grid_size  = (n + block_size - 1) / block_size;
    GenerateKernel<<<grid_size, block_size>>>(
        d_values_flat, d_offsets, d_lengths,
        n, d_prefix, prefix_len,
        d_out_flat, max_guess_len);
    cudaDeviceSynchronize();

    // ── 6. D→H 拷贝结果 ──
    std::vector<char> h_out((size_t)n * max_guess_len);
    cudaMemcpy(h_out.data(), d_out_flat, (size_t)n * max_guess_len, cudaMemcpyDeviceToHost);

    // ── 7. 将扁平缓冲区转回 vector<string> ──
    out_guesses.reserve(out_guesses.size() + n);
    for (int i = 0; i < n; ++i) {
        const char* s = h_out.data() + (size_t)i * max_guess_len;
        out_guesses.emplace_back(s);
    }
    total_guesses += n;

    // ── 8. 释放 device 内存 ──
    cudaFree(d_values_flat);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_prefix);
    cudaFree(d_out_flat);
}
