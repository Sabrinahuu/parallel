#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <cstring>
using namespace std;
using namespace chrono;

// ============================================================
// 序列化工具
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
static void pack_pt(vector<char> &buf, const PT &pt)
{
    pack_int(buf, pt.pivot);
    pack_float(buf, pt.preterm_prob);
    pack_float(buf, pt.prob);
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
// 广播整个模型：rank0训练完后序列化广播给所有worker
// ============================================================
static void BcastModel(model &m, int root)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<char> buf;

    if (rank == root)
    {
        pack_int(buf, m.total_preterm);

        pack_int(buf, m.letters.size());
        for (const segment &s : m.letters) pack_segment_full(buf, s);

        pack_int(buf, m.digits.size());
        for (const segment &s : m.digits) pack_segment_full(buf, s);

        pack_int(buf, m.symbols.size());
        for (const segment &s : m.symbols) pack_segment_full(buf, s);

        pack_int(buf, m.preterm_freq.size());
        for (auto &kv : m.preterm_freq) { pack_int(buf, kv.first); pack_int(buf, kv.second); }

        pack_int(buf, m.ordered_pts.size());
        for (const PT &pt : m.ordered_pts) pack_pt(buf, pt);

        pack_int(buf, m.preterminals.size());
        for (const PT &pt : m.preterminals) pack_pt(buf, pt);
    }

    int bufsz = buf.size();
    MPI_Bcast(&bufsz, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank != root) buf.resize(bufsz);
    MPI_Bcast(buf.data(), bufsz, MPI_CHAR, root, MPI_COMM_WORLD);

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

    // MD5测试只让rank==0做
    if (rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;
        string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
        string test_hashes[8] = {
            "e10adc3949ba59abbe56e057f20f883e",
            "5f4dcc3b5aa765d61d8327deb882cf99",
            "25d55ad283aa400af464c76d713c07ad",
            "d8578edf8458ce06fbc5bb76a58c5ca4",
            "25f9e794323b453885f5181f1b624d0b",
            "827ccb0eea8a706c4c34a16891f84e7b",
            "81dc9bdb52d04dc20036dbd8313ed055",
            "96e79218965eb72c92a549dd5a330112"
        };
        for (int i = 0; i < 8; i++)
        {
            bit32 state[4];
            MD5Hash(test_pws[i], state);
            stringstream ss;
            for (int i1 = 0; i1 < 4; i1++)
                ss << std::setw(8) << std::setfill('0') << hex << state[i1];
            if (ss.str() != test_hashes[i])
            {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        cout << "MD5Hash test passed!" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    // -------------------------------------------------------
    // 训练阶段：只让rank==0训练，完成后广播模型给所有worker
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
        q.init();
        cout << "here" << endl;
        int last_report = 0;
        auto start = system_clock::now();

        while (!q.priority.empty())
        {
            int flag = 1;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

            q.PopNext();

            if (q.total_guesses - last_report >= 100000)
            {
                cout << "Guesses generated: " << q.total_guesses << endl;
                last_report = q.total_guesses;

                if (q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    time_hash = q.hash_time;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    break;
                }
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
