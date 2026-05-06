#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include <array>
#include <sstream>
#include <iostream>

using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    // 原始输入字符串
    string input = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";

    // 为 NEON batch 版本准备 4 路相同输入
    vector<string> inputs = {input, input, input, input};
    vector<array<bit32, 4>> outputs;

    // 调用 SIMD 版本
    MD5HashBatch_NEON(inputs, &outputs);

    cout << "SIMD结果: ";
    for (int i = 0; i < 4; i++)
    {
        cout << setw(8) << setfill('0') << hex << outputs[0][i];
    }
    cout << endl;

    // 调用原始标量版本
    bit32 state[4];
    MD5Hash(input, state);

    cout << "原始MD5Hash结果: ";
    for (int i = 0; i < 4; i++)
    {
        cout << setw(8) << setfill('0') << hex << state[i];
    }
    cout << endl;

    // 验证两个结果是否相同
    bool match = true;
    for (int i = 0; i < 4; i++)
    {
        if (state[i] != outputs[0][i])
        {
            match = false;
            break;
        }
    }

    if (match)
    cout << "NEONMD5 has passed!" << endl;
    else
    cout << "NEONMD5 has failed." << endl;

    return 0;
}