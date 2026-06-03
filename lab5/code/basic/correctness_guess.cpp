#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
#include <cstring>
using namespace std;
using namespace chrono;

// ============================================================
// 序列化工具：把数据写入/读出 vector<char> 缓冲区
// ============================================================
static void pack_int(vector<char> &buf, int v)
{
    const char *p = reinterpret_cast<const char*>(&v);
    buf.insert(buf.end(), p, p + sizeof(int));
}
static void pack_float(vector<char> &buf, float v)
{
    const char *p = reinterpret_cast<const char*>(&v);
    buf.insert(buf.end(), p, p + sizeof(float));
}
static void pack_string(vector<char> &buf, const string &s)
{
    int len = s.size();
    pack_int(buf, len);
    buf.insert(buf.end(), s.begin(), s.end());
}
static void pack_intvec(vector<char> &buf, const vector<int> &v)
{
    pack_int(buf, v.size());
    for (int x : v) pack_int(buf, x);
}
static void pack_strvec(vector<char> &buf, const vector<string> &v)
{
    pack_int(buf, v.size());
    for (const string &s : v) pack_string(buf, s);
}

static int unpack_int(const vector<char> &buf, int &off)
{
    int v; memcpy(&v, buf.data() + off, sizeof(int)); off += sizeof(int); return v;
}
static float unpack_float(const vector<char> &buf, int &off)
{
    float v; memcpy(&v, buf.data() + off, sizeof(float)); off += sizeof(float); return v;
}
static string unpack_string(const vector<char> &buf, int &off)
{
    int len = unpack_int(buf, off);
    string s(buf.data() + off, len); off += len; return s;
}
static vector<int> unpack_intvec(const vector<char> &buf, int &off)
{
    int sz = unpack_int(buf, off);
    vector<int> v(sz);
    for (int i = 0; i < sz; i++) v[i] = unpack_int(buf, off);
    return v;
}
static vector<string> unpack_strvec(const vector<char> &buf, int &off)
{
    int sz = unpack_int(buf, off);
    vector<string> v(sz);
    for (int i = 0; i < sz; i++) v[i] = unpack_string(buf, off);
    return v;
}

// ============================================================
// 序列化 segment（完整字段，供letters/digits/symbols使用）
// ============================================================
static void pack_segment_full(vector<char> &buf, const segment &seg)
{
    pack_int(buf, seg.type);
    pack_int(buf, seg.length);
    pack_int(buf, seg.total_freq);
    pack_strvec(buf, seg.ordered_values);
    pack_intvec(buf, seg.ordered_freqs);
}
static segment unpack_segment_full(const vector<char> &buf, int &off)
{
    int type   = unpack_int(buf, off);
    int length = unpack_int(buf, off);
    segment seg(type, length);
    seg.total_freq     = unpack_int(buf, off);
    seg.ordered_values = unpack_strvec(buf, off);
    seg.ordered_freqs  = unpack_intvec(buf, off);
    return seg;
}

// ============================================================
// 序列化 segment（只有type/length，供PT::content使用）
// ============================================================
static void pack_segment_meta(vector<char> &buf, const segment &seg)
{
    pack_int(buf, seg.type);
    pack_int(buf, seg.length);
}
static segment unpack_segment_meta(const vector<char> &buf, int &off)
{
    int type   = unpack_int(buf, off);
    int length = unpack_int(buf, off);
    return segment(type, length);
}

// ============================================================
// 序列化 PT
// ============================================================
static void pack_pt(vector<char> &buf, const PT &pt)
{
    pack_int(buf, pt.pivot);
    pack_float(buf, pt.preterm_prob);
    pack_float(buf, pt.prob);
    // content：只需type/length
    pack_int(buf, pt.content.size());
    for (const segment &seg : pt.content) pack_segment_meta(buf, seg);
    pack_intvec(buf, pt.curr_indices);
    pack_intvec(buf, pt.max_indices);
}
static PT unpack_pt(const vector<char> &buf, int &off)
{
    PT pt;
    pt.pivot        = unpack_int(buf, off);
    pt.preterm_prob = unpack_float(buf, off);
    pt.prob         = unpack_float(buf, off);
    int csz = unpack_int(buf, off);
    for (int i = 0; i < csz; i++) pt.content.push_back(unpack_segment_meta(buf, off));
    pt.curr_indices = unpack_intvec(buf, off);
    pt.max_indices  = unpack_intvec(buf, off);
    return pt;
}

// ============================================================
// 序列化整个model，广播给所有进程
// ============================================================
static void BcastModel(model &m, int root)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<char> buf;

    if (rank == root)
    {
        // total_preterm
        pack_int(buf, m.total_preterm);

        // letters
        pack_int(buf, m.letters.size());
        for (const segment &s : m.letters) pack_segment_full(buf, s);

        // digits
        pack_int(buf, m.digits.size());
        for (const segment &s : m.digits) pack_segment_full(buf, s);

        // symbols
        pack_int(buf, m.symbols.size());
        for (const segment &s : m.symbols) pack_segment_full(buf, s);

        // preterm_freq
        pack_int(buf, m.preterm_freq.size());
        for (auto &kv : m.preterm_freq) { pack_int(buf, kv.first); pack_int(buf, kv.second); }

        // ordered_pts
        pack_int(buf, m.ordered_pts.size());
        for (const PT &pt : m.ordered_pts) pack_pt(buf, pt);

        // preterminals
        pack_int(buf, m.preterminals.size());
        for (const PT &pt : m.preterminals) pack_pt(buf, pt);
    }

    // 先广播缓冲区大小，再广播内容
    int bufsz = buf.size();
    MPI_Bcast(&bufsz, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank != root) buf.resize(bufsz);
    MPI_Bcast(buf.data(), bufsz, MPI_CHAR, root, MPI_COMM_WORLD);

    // worker端反序列化
    if (rank != root)
    {
        int off = 0;

        m.total_preterm = unpack_int(buf, off);

        int lsz = unpack_int(buf, off);
        for (int i = 0; i < lsz; i++) m.letters.push_back(unpack_segment_full(buf, off));

        int dsz = unpack_int(buf, off);
        for (int i = 0; i < dsz; i++) m.digits.push_back(unpack_segment_full(buf, off));

        int ssz = unpack_int(buf, off);
        for (int i = 0; i < ssz; i++) m.symbols.push_back(unpack_segment_full(buf, off));

        int pfsz = unpack_int(buf, off);
        for (int i = 0; i < pfsz; i++)
        {
            int k = unpack_int(buf, off);
            int v = unpack_int(buf, off);
            m.preterm_freq[k] = v;
        }

        int optsz = unpack_int(buf, off);
        for (int i = 0; i < optsz; i++) m.ordered_pts.push_back(unpack_pt(buf, off));

        int ptsz = unpack_int(buf, off);
        for (int i = 0; i < ptsz; i++) m.preterminals.push_back(unpack_pt(buf, off));
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    // -------------------------------------------------------
    // 训练阶段：只让 rank==0 读取训练集并构建模型
    // 训练完成后序列化广播给所有 worker
    // -------------------------------------------------------
    auto start_train = system_clock::now();
    if (rank == 0)
    {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt", 0);
        q.m.order(0);
    }
    BcastModel(q.m, 0);
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    if (rank == 0)
    {
        unordered_set<std::string> test_set;
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while (test_data >> pw)
        {
            test_count += 1;
            test_set.insert(pw);
            if (test_count >= 1000000) break;
        }
        int cracked = 0;

        q.init();
        cout << "here" << endl;
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;

        while (!q.priority.empty())
        {
            int flag = 1;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

            q.PopNext();
            q.total_guesses = q.guesses.size();

            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                if (history + q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    cout << "Cracked:" << cracked << endl;
                    break;
                }
            }

            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    if (test_set.find(pw) != test_set.end()) cracked += 1;
                    MD5Hash(pw, state);
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        int flag = 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        while (true)
        {
            int flag;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (flag == 0) break;
            q.GenerateWorker();
        }
    }

    MPI_Finalize();
    return 0;
}