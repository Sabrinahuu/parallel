// guesses_gpu.h
#pragma once
#include <string>
#include <vector>

struct BatchTask {
    std::string prefix;
    std::vector<std::string> values;
};

void GenerateBatchOnGPU(
    const std::vector<BatchTask>& tasks,
    std::vector<std::string>&     out_guesses,
    int&                          total_guesses);
