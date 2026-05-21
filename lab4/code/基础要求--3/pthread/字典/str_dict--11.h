#pragma once

#include "PCFG.h"
#include <vector>
#include <string>
#include <cstring>

// 支持的最大字符串长度（密码长度上限通常为 32，留余量取 64）
static const int DICT_MAX_LEN = 64;
// 类型数：1=字母, 2=数字, 3=符号，下标直接用 1/2/3，0 不用
static const int DICT_MAX_TYPE = 4;

class StrDict {
public:
    // dict_data：真正存字符串内容的连续内存
    std::vector<char> dict_data;

    // start_pos[type][length]：该 (type,length) 组合在 dict_data 中的起始字节偏移
    // 若不存在该组合，值为 -1
    int start_pos[DICT_MAX_TYPE][DICT_MAX_LEN];

    // count[type][length]：该 (type,length) 组合共有多少个字符串（即 ordered_values.size()）
    int count_arr[DICT_MAX_TYPE][DICT_MAX_LEN];

    StrDict() {
        memset(start_pos, -1, sizeof(start_pos));
        memset(count_arr, 0,  sizeof(count_arr));
    }

    
    void Build(const model &m) {
        dict_data.clear();
        memset(start_pos, -1, sizeof(start_pos));
        memset(count_arr, 0,  sizeof(count_arr));

        // 按 type=1(字母), 2(数字), 3(符号) 依次处理
        // segs[0] 对应 type=1，segs[1] 对应 type=2，segs[2] 对应 type=3
        const std::vector<segment>* segs[3] = {&m.letters, &m.digits, &m.symbols};

        for (int type = 1; type <= 3; ++type) {
            const std::vector<segment> &seg_list = *segs[type - 1];
            for (const segment &seg : seg_list) {
                int len = seg.length;
                if (len <= 0 || len >= DICT_MAX_LEN) continue;
                if (seg.ordered_values.empty()) continue;

                // 记录起始偏移（当前 dict_data 末尾）
                start_pos[type][len] = (int)dict_data.size();
                count_arr[type][len] = (int)seg.ordered_values.size();

                // 按 rank 顺序（ordered_values 已是概率降序）写入每个字符串
                for (const std::string &val : seg.ordered_values) {
                    // 写入 len 个字节；若实际长度不足则补 '\0'（理论上不会发生）
                    for (int c = 0; c < len; ++c) {
                        dict_data.push_back(c < (int)val.size() ? val[c] : '\0');
                    }
                }
            }
        }
    }

    // Lookup：返回 type/length/rank 对应字符串的起始指针（长度为 length）
    //         调用方自行用 string(ptr, length) 构造 std::string。
    //         若参数非法或不存在，返回 nullptr。
    inline const char* Lookup(int type, int length, int rank) const {
        if (type < 1 || type >= DICT_MAX_TYPE) return nullptr;
        if (length <= 0 || length >= DICT_MAX_LEN) return nullptr;
        int sp = start_pos[type][length];
        if (sp < 0) return nullptr;
        if (rank < 0 || rank >= count_arr[type][length]) return nullptr;
        // position = starting_position + rank * length  （论文公式）
        return dict_data.data() + sp + rank * length;
    }


    // Count：返回 (type, length) 下的字符串总数，不存在则返回 0
    inline int Count(int type, int length) const {
        if (type < 1 || type >= DICT_MAX_TYPE) return 0;
        if (length <= 0 || length >= DICT_MAX_LEN) return 0;
        return count_arr[type][length];
    }


    // LookupString：直接返回 std::string，方便在 Generate 中使用
    inline std::string LookupString(int type, int length, int rank) const {
        const char* p = Lookup(type, length, rank);
        if (!p) return "";
        return std::string(p, length);
    }
};