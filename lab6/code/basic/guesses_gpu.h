// guesses_gpu.h
#pragma once
#include <string>
#include <vector>

// GPU 并行生成口令猜测。
// values       : 最后一个 segment 在模型中的所有 ordered_values
// prefix       : 已拼好的前缀字符串（单 segment PT 时传入空字符串）
// out_guesses  : 输出列表（追加写入）
// total_guesses: 全局计数（累加写入）
void GenerateOnGPU(
    const std::vector<std::string>& values,
    const std::string&              prefix,
    std::vector<std::string>&       out_guesses,
    int&                            total_guesses);