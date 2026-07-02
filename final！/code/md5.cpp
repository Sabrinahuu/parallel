#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>

#include <arm_neon.h>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
    Byte *blocks = (Byte *)input.c_str();
    int length = input.length();
    int bitLength = length * 8;

    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
        paddingBits = 512 - (paddingBits - 448);
    else if (paddingBits < 448)
        paddingBits = 448 - paddingBits;
    else
        paddingBits = 512;

    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    memcpy(paddedMessage, blocks, length);
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);

    for (int i = 0; i < 8; ++i)
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;

    *n_byte = paddedLength;
    return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */ 
void MD5Hash(string input, bit32 *state)
{
    Byte *paddedMessage;
    int *messageLength = new int[1];
    for (int i = 0; i < 1; i += 1)
    {
        paddedMessage = StringProcess(input, &messageLength[i]);
        assert(messageLength[i] == messageLength[0]);
    }
    int n_blocks = messageLength[0] / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    for (int i = 0; i < n_blocks; i += 1)
    {
        bit32 x[16];
        for (int i1 = 0; i1 < 16; ++i1)
        {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                     (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                     (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                     (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);

        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |
                   ((value & 0xff00) << 8) |
                   ((value & 0xff0000) >> 8) |
                   ((value & 0xff000000) >> 24);
    }

    delete[] paddedMessage;
    delete[] messageLength;
}


// NEON 4-lane batch MD5 

//32位字节序翻转
static inline bit32 bswap32_md5(bit32 v)
{
    return __builtin_bswap32(v);
}

//NEON版本的循环左移
#define NEON_ROL(x, n) \
    vorrq_u32(vshlq_n_u32((x), (n)), vshrq_n_u32((x), 32 - (n)))

//向量版本的MD5的四个布尔函数
#define F4(x, y, z) vbslq_u32((x), (y), (z))
#define G4(x, y, z) vbslq_u32((z), (x), (y))
#define H4(x, y, z) veorq_u32(veorq_u32((x), (y)), (z))
#define I4(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32((z))))



//针对短消息（长度小于65）的padding函数
static __attribute__((always_inline)) inline int
PadShort(const char *data, int length, Byte *out)
{
    memcpy(out, data, length);
    out[length] = 0x80;
    memset(out + length + 1, 0, 55 - length);
    uint64_t bitLen = (uint64_t)length * 8;
    memcpy(out + 56, &bitLen, 8);
    return 1; /* 1 block */
}

//通用padding：适用消息较长、一个block放不下的情况
static __attribute__((always_inline)) inline int
PadGeneral(const char *data, int length, Byte *out)
{
    int bitLength = length * 8;
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
        paddingBits = 512 - (paddingBits - 448);
    else if (paddingBits < 448)
        paddingBits = 448 - paddingBits;
    else
        paddingBits = 512;

    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;

    memcpy(out, data, length);
    out[length] = 0x80;
    memset(out + length + 1, 0, paddingBytes - 1);

    uint64_t bitLen64 = (uint64_t)length * 8;
    memcpy(out + length + paddingBytes, &bitLen64, 8);

    return paddedLength / 64;
}

//NEON版本的StringProcess
static __attribute__((always_inline)) inline int
StringProcessFlat(const string &input, Byte *out)
{
    int length = (int)input.length();
    if (length < 56)
        return PadShort(input.c_str(), length, out);
    return PadGeneral(input.c_str(), length, out);
}



//全0的假block
static const Byte ZERO_BLOCK[64] = {0};

/* ================================================================== */
/*  MD5HashBatch_NEON：NEON的主函数                                                  */
/* ================================================================== */

void MD5HashBatch_NEON(const vector<string> &inputs, vector<array<bit32, 4>> *outputs)
{
    if (outputs != nullptr)
        outputs->resize(inputs.size());

    alignas(64) static Byte padBuf[4][8192];

    //每次取最多4条消息作为一个batch
    for (size_t base = 0; base < inputs.size(); base += 4)
    {   
        //统计真实消息数目
        int valid = (int)std::min<size_t>(4, inputs.size() - base);

        int n_blocks[4];
        
        //对真实消息做padding
        for (int lane = 0; lane < valid; ++lane)
            n_blocks[lane] = StringProcessFlat(inputs[base + lane], padBuf[lane]);
        
        //补齐无效lane
        for (int lane = valid; lane < 4; ++lane)
        {
            padBuf[lane][0] = 0x80;
            memset(padBuf[lane] + 1, 0, 63);
            n_blocks[lane] = 1;
        }
        
        //求最大block数
        int max_blocks = n_blocks[0];
        for (int lane = 1; lane < 4; ++lane)
            if (n_blocks[lane] > max_blocks)
                max_blocks = n_blocks[lane];
        
        //初始化4路MD5状态
        uint32x4_t A_state = vdupq_n_u32(0x67452301);
        uint32x4_t B_state = vdupq_n_u32(0xefcdab89);
        uint32x4_t C_state = vdupq_n_u32(0x98badcfe);
        uint32x4_t D_state = vdupq_n_u32(0x10325476);

        for (int blk = 0; blk < max_blocks; ++blk)
        {
            const Byte *b0 = (blk < n_blocks[0]) ? (padBuf[0] + blk * 64) : ZERO_BLOCK;
            const Byte *b1 = (blk < n_blocks[1]) ? (padBuf[1] + blk * 64) : ZERO_BLOCK;
            const Byte *b2 = (blk < n_blocks[2]) ? (padBuf[2] + blk * 64) : ZERO_BLOCK;
            const Byte *b3 = (blk < n_blocks[3]) ? (padBuf[3] + blk * 64) : ZERO_BLOCK;

            const uint32_t *w0 = reinterpret_cast<const uint32_t *>(b0);
            const uint32_t *w1 = reinterpret_cast<const uint32_t *>(b1);
            const uint32_t *w2 = reinterpret_cast<const uint32_t *>(b2);
            const uint32_t *w3 = reinterpret_cast<const uint32_t *>(b3);

//把4个lane相同的word打包成一个向量
#define LOAD_X(name, k)                                                  \
    alignas(16) uint32_t name##_tmp[4] = { w0[k], w1[k], w2[k], w3[k] }; \
    uint32x4_t name = vld1q_u32(name##_tmp)

            /* 预装载 16 个 message word */
            LOAD_X(X0,  0);
            LOAD_X(X1,  1);
            LOAD_X(X2,  2);
            LOAD_X(X3,  3);
            LOAD_X(X4,  4);
            LOAD_X(X5,  5);
            LOAD_X(X6,  6);
            LOAD_X(X7,  7);
            LOAD_X(X8,  8);
            LOAD_X(X9,  9);
            LOAD_X(X10, 10);
            LOAD_X(X11, 11);
            LOAD_X(X12, 12);
            LOAD_X(X13, 13);
            LOAD_X(X14, 14);
            LOAD_X(X15, 15);

            uint32x4_t a = A_state;
            uint32x4_t b = B_state;
            uint32x4_t c = C_state;
            uint32x4_t d = D_state;

#define FSTEP(a,b,c,d,x,s,ti)                                             \
    do {                                                                   \
        uint32x4_t _t = vaddq_u32((a), F4((b),(c),(d)));                   \
        _t = vaddq_u32(_t, vaddq_u32((x), vdupq_n_u32((ti))));             \
        (a) = vaddq_u32((b), NEON_ROL(_t, (s)));                           \
    } while (0)

#define GSTEP(a,b,c,d,x,s,ti)                                             \
    do {                                                                   \
        uint32x4_t _t = vaddq_u32((a), G4((b),(c),(d)));                   \
        _t = vaddq_u32(_t, vaddq_u32((x), vdupq_n_u32((ti))));             \
        (a) = vaddq_u32((b), NEON_ROL(_t, (s)));                           \
    } while (0)

#define HSTEP(a,b,c,d,x,s,ti)                                             \
    do {                                                                   \
        uint32x4_t _t = vaddq_u32((a), H4((b),(c),(d)));                   \
        _t = vaddq_u32(_t, vaddq_u32((x), vdupq_n_u32((ti))));             \
        (a) = vaddq_u32((b), NEON_ROL(_t, (s)));                           \
    } while (0)

#define ISTEP(a,b,c,d,x,s,ti)                                             \
    do {                                                                   \
        uint32x4_t _t = vaddq_u32((a), I4((b),(c),(d)));                   \
        _t = vaddq_u32(_t, vaddq_u32((x), vdupq_n_u32((ti))));             \
        (a) = vaddq_u32((b), NEON_ROL(_t, (s)));                           \
    } while (0)

            /* ================= Round 1 ================= */
            FSTEP(a,b,c,d,X0,  7, 0xd76aa478);
            FSTEP(d,a,b,c,X1, 12, 0xe8c7b756);
            FSTEP(c,d,a,b,X2, 17, 0x242070db);
            FSTEP(b,c,d,a,X3, 22, 0xc1bdceee);
            FSTEP(a,b,c,d,X4,  7, 0xf57c0faf);
            FSTEP(d,a,b,c,X5, 12, 0x4787c62a);
            FSTEP(c,d,a,b,X6, 17, 0xa8304613);
            FSTEP(b,c,d,a,X7, 22, 0xfd469501);
            FSTEP(a,b,c,d,X8,  7, 0x698098d8);
            FSTEP(d,a,b,c,X9, 12, 0x8b44f7af);
            FSTEP(c,d,a,b,X10,17, 0xffff5bb1);
            FSTEP(b,c,d,a,X11,22, 0x895cd7be);
            FSTEP(a,b,c,d,X12, 7, 0x6b901122);
            FSTEP(d,a,b,c,X13,12, 0xfd987193);
            FSTEP(c,d,a,b,X14,17, 0xa679438e);
            FSTEP(b,c,d,a,X15,22, 0x49b40821);

            /* ================= Round 2 ================= */
            GSTEP(a,b,c,d,X1,  5, 0xf61e2562);
            GSTEP(d,a,b,c,X6,  9, 0xc040b340);
            GSTEP(c,d,a,b,X11,14, 0x265e5a51);
            GSTEP(b,c,d,a,X0, 20, 0xe9b6c7aa);
            GSTEP(a,b,c,d,X5,  5, 0xd62f105d);
            GSTEP(d,a,b,c,X10, 9, 0x02441453);
            GSTEP(c,d,a,b,X15,14, 0xd8a1e681);
            GSTEP(b,c,d,a,X4, 20, 0xe7d3fbc8);
            GSTEP(a,b,c,d,X9,  5, 0x21e1cde6);
            GSTEP(d,a,b,c,X14, 9, 0xc33707d6);
            GSTEP(c,d,a,b,X3, 14, 0xf4d50d87);
            GSTEP(b,c,d,a,X8, 20, 0x455a14ed);
            GSTEP(a,b,c,d,X13, 5, 0xa9e3e905);
            GSTEP(d,a,b,c,X2,  9, 0xfcefa3f8);
            GSTEP(c,d,a,b,X7, 14, 0x676f02d9);
            GSTEP(b,c,d,a,X12,20, 0x8d2a4c8a);

            /* ================= Round 3 ================= */
            HSTEP(a,b,c,d,X5,  4, 0xfffa3942);
            HSTEP(d,a,b,c,X8, 11, 0x8771f681);
            HSTEP(c,d,a,b,X11,16, 0x6d9d6122);
            HSTEP(b,c,d,a,X14,23, 0xfde5380c);
            HSTEP(a,b,c,d,X1,  4, 0xa4beea44);
            HSTEP(d,a,b,c,X4, 11, 0x4bdecfa9);
            HSTEP(c,d,a,b,X7, 16, 0xf6bb4b60);
            HSTEP(b,c,d,a,X10,23, 0xbebfbc70);
            HSTEP(a,b,c,d,X13, 4, 0x289b7ec6);
            HSTEP(d,a,b,c,X0, 11, 0xeaa127fa);
            HSTEP(c,d,a,b,X3, 16, 0xd4ef3085);
            HSTEP(b,c,d,a,X6, 23, 0x04881d05);
            HSTEP(a,b,c,d,X9,  4, 0xd9d4d039);
            HSTEP(d,a,b,c,X12,11, 0xe6db99e5);
            HSTEP(c,d,a,b,X15,16, 0x1fa27cf8);
            HSTEP(b,c,d,a,X2, 23, 0xc4ac5665);

            /* ================= Round 4 ================= */
            ISTEP(a,b,c,d,X0,  6, 0xf4292244);
            ISTEP(d,a,b,c,X7, 10, 0x432aff97);
            ISTEP(c,d,a,b,X14,15, 0xab9423a7);
            ISTEP(b,c,d,a,X5, 21, 0xfc93a039);
            ISTEP(a,b,c,d,X12, 6, 0x655b59c3);
            ISTEP(d,a,b,c,X3, 10, 0x8f0ccc92);
            ISTEP(c,d,a,b,X10,15, 0xffeff47d);
            ISTEP(b,c,d,a,X1, 21, 0x85845dd1);
            ISTEP(a,b,c,d,X8,  6, 0x6fa87e4f);
            ISTEP(d,a,b,c,X15,10, 0xfe2ce6e0);
            ISTEP(c,d,a,b,X6, 15, 0xa3014314);
            ISTEP(b,c,d,a,X13,21, 0x4e0811a1);
            ISTEP(a,b,c,d,X4,  6, 0xf7537e82);
            ISTEP(d,a,b,c,X11,10, 0xbd3af235);
            ISTEP(c,d,a,b,X2, 15, 0x2ad7d2bb);
            ISTEP(b,c,d,a,X9, 21, 0xeb86d391);

            /* 按 lane 有效性写回，避免不同消息长度时错误累加 */
            alignas(16) uint32_t m_arr[4] = {
                (blk < n_blocks[0]) ? 0xFFFFFFFFu : 0u,
                (blk < n_blocks[1]) ? 0xFFFFFFFFu : 0u,
                (blk < n_blocks[2]) ? 0xFFFFFFFFu : 0u,
                (blk < n_blocks[3]) ? 0xFFFFFFFFu : 0u
            };
            uint32x4_t active = vld1q_u32(m_arr);
            
            //选择性更新状态
            uint32x4_t newA = vaddq_u32(A_state, a);
            uint32x4_t newB = vaddq_u32(B_state, b);
            uint32x4_t newC = vaddq_u32(C_state, c);
            uint32x4_t newD = vaddq_u32(D_state, d);

            A_state = vbslq_u32(active, newA, A_state);
            B_state = vbslq_u32(active, newB, B_state);
            C_state = vbslq_u32(active, newC, C_state);
            D_state = vbslq_u32(active, newD, D_state);

#undef LOAD_X
#undef FSTEP
#undef GSTEP
#undef HSTEP
#undef ISTEP
        }

        if (outputs != nullptr)
        {
            uint32_t A_arr[4], B_arr[4], C_arr[4], D_arr[4];
            vst1q_u32(A_arr, A_state);
            vst1q_u32(B_arr, B_state);
            vst1q_u32(C_arr, C_state);
            vst1q_u32(D_arr, D_state);

            for (int lane = 0; lane < valid; ++lane)
            {
                (*outputs)[base + lane] = {
                    bswap32_md5(A_arr[lane]),
                    bswap32_md5(B_arr[lane]),
                    bswap32_md5(C_arr[lane]),
                    bswap32_md5(D_arr[lane])
                };
            }
        }
    }
}