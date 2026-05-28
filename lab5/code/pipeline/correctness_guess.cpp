#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <string>
using namespace std;
using namespace chrono;

// 缂栬瘧鎸囦护绀轰緥锛�
// g++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess -O2

int main()
{
    const long long MAX_GUESSES = 10000000;      // 鐢熸垚鍙ｄ护鎬讳笂闄�
    const long long REPORT_INTERVAL = 100000;    // 姣忕敓鎴愬灏戞潯杈撳嚭涓€娆�
    const size_t BATCH_SIZE = 1000000;           // 姣忔壒澶勭悊澶氬皯鏉★紝閬垮厤鍐呭瓨杩囧ぇ
    const int TEST_SET_LIMIT = 1000000;          // 鍙栧墠100涓囨潯鍙ｄ护浣滀负娴嬭瘯闆嗗悎

    double time_hash = 0.0;   // MD5鍝堝笇鑰楁椂
    double time_guess = 0.0;  // 绾敓鎴愯€楁椂 = 鎬昏€楁椂 - 鍝堝笇鑰楁椂
    double time_train = 0.0;  // 妯″瀷璁粌鑰楁椂

    PriorityQueue q;

    // ============================================================
    // 1. 璁粌 PCFG 妯″瀷
    // ============================================================
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();

    time_train = double(duration_cast<microseconds>(end_train - start_train).count())
                 * microseconds::period::num / microseconds::period::den;

    // ============================================================
    // 2. 鍔犺浇鍓� TEST_SET_LIMIT 鏉＄湡瀹炲彛浠わ紝浣滀负 correctness 娴嬭瘯闆嗗悎
    // ============================================================
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");

    if (!test_data.is_open())
    {
        cerr << "Failed to open /guessdata/Rockyou-singleLined-full.txt" << endl;
        return 1;
    }

    int test_count = 0;
    string pw;
    while (test_data >> pw)
    {
        test_set.insert(pw);
        test_count++;

        if (test_count >= TEST_SET_LIMIT)
            break;
    }

    int cracked = 0;

    // ============================================================
    // 3. 涓茶鐢熸垚 + 涓茶鍝堝笇锛岀敤浣滄纭€у熀鍑�
    //
    // 淇敼鐐癸細
    //   1锛変笉鍐嶄娇鐢ㄦ湭鐢熸晥鐨� generate_n 灞€閮ㄥ彉閲忥紱
    //   2锛夌敤 q.guesses.size() 鍒ゆ柇鏄惁杈惧埌鎵规澶у皬锛�
    //   3锛夎揪鍒� MAX_GUESSES 鏃朵細鎴柇鏈€鍚庝竴鎵癸紝閬垮厤鏄庢樉瓒呭嚭涓婇檺锛�
    //   4锛夊惊鐜粨鏉熸椂浼氬鐞嗘渶鍚庝笉瓒充竴鎵圭殑鍓╀綑鍙ｄ护锛�
    //   5锛夎緭鍑� total guesses processed锛屾柟渚垮拰 MPI 鐗堟湰瀵归綈銆�
    // ============================================================
    q.init();
    cout << "Start correctness guessing..." << endl;

    long long history = 0;       // 宸茬粡澶勭悊杩囩殑鍙ｄ护鏁伴噺
    long long last_report = 0;   // 涓婁竴娆¤繘搴﹁緭鍑轰綅缃�

    auto start = system_clock::now();

    auto process_batch = [&](vector<string>& guesses)
    {
        if (guesses.empty())
            return;

        auto start_hash = system_clock::now();

        bit32 state[4];

        for (const string& candidate : guesses)
        {
            if (test_set.find(candidate) != test_set.end())
                cracked++;

            MD5Hash(candidate, state);
        }

        auto end_hash = system_clock::now();
        time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                     * microseconds::period::num / microseconds::period::den;

        history += static_cast<long long>(guesses.size());
        guesses.clear();
    };

    while (!q.priority.empty() && history < MAX_GUESSES)
    {
        q.PopNext();

        long long current_total = history + static_cast<long long>(q.guesses.size());

        // 杩涘害杈撳嚭
        while (current_total - last_report >= REPORT_INTERVAL)
        {
            last_report += REPORT_INTERVAL;
            cout << "Guesses generated: " << last_report << endl;
        }

        // 杈惧埌鐢熸垚涓婇檺鏃讹紝鎴柇褰撳墠鎵规锛屽鐞嗗悗閫€鍑�
        if (current_total >= MAX_GUESSES)
        {
            size_t keep_num = static_cast<size_t>(MAX_GUESSES - history);

            if (q.guesses.size() > keep_num)
                q.guesses.resize(keep_num);

            process_batch(q.guesses);
            break;
        }

        // 杈惧埌鎵规澶у皬灏卞鐞嗭紝閬垮厤 q.guesses 鎸佺画鑶ㄨ儉
        if (q.guesses.size() >= BATCH_SIZE)
        {
            process_batch(q.guesses);
        }
    }

    // 濡傛灉浼樺厛闃熷垪鑰楀敖鏃惰繕鏈変笉瓒充竴鎵圭殑鍓╀綑鍙ｄ护锛屼篃瑕佸鐞�
    if (!q.guesses.empty() && history < MAX_GUESSES)
    {
        process_batch(q.guesses);
    }

    auto end = system_clock::now();
    double total_time = double(duration_cast<microseconds>(end - start).count())
                        * microseconds::period::num / microseconds::period::den;

    time_guess = total_time - time_hash;

    cout << "Total guesses processed: " << history << endl;
    cout << "Guess time:" << time_guess << "seconds" << endl;
    cout << "Hash time:" << time_hash << "seconds" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;
    cout << "Cracked:" << cracked << endl;

    return 0;
}