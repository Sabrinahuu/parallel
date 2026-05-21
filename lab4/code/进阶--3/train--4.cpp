#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <omp.h>



// ---------- 将 src segment 的 value 频率合并进 dst ----------
static void mergeSegValues(segment &dst, const segment &src)
{
    for (const auto &kv : src.values)
    {
        const string &val   = kv.first;//value字符串
        int           s_id  = kv.second;//该value在src中的编号
        int           sfreq = src.freqs.at(s_id);//该value在src中的频率

        auto it = dst.values.find(val);
        if (it == dst.values.end())//如果dst中没有这个value
        {
            int new_id        = (int)dst.values.size();//分配新编号
            dst.values[val]   = new_id;
            dst.freqs[new_id] = sfreq;//直接写入频率
        }
        else //dst中已有这个value
        {
            dst.freqs[it->second] += sfreq;
        }
    }
}

//分四块合并
void model::mergeFrom(const model &other)
{
    total_preterm += other.total_preterm;

    // ---- letters ----
    for (int i = 0; i < (int)other.letters.size(); i++)
    {
        const segment &s = other.letters[i];
        int id = FindLetter(s);//在当前模型里按length查找
        if (id == -1)//当前模型没有这个长度的letter segment
        {
            id = GetNextLettersID();
            // 只保留 type/length，values/freqs 通过 mergeSegValues 写入
            segment ns(s.type, s.length);
            letters.emplace_back(ns);
            letters_freq[id] = other.letters_freq.at(i);//复制频率
            mergeSegValues(letters[id], s);//合并values
        }
        else  //已经存在就直接累加
        {
            letters_freq[id] += other.letters_freq.at(i);
            mergeSegValues(letters[id], s);
        }
    }

    // ---- digits ----
    for (int i = 0; i < (int)other.digits.size(); i++)
    {
        const segment &s = other.digits[i];
        int id = FindDigit(s);
        if (id == -1)
        {
            id = GetNextDigitsID();
            segment ns(s.type, s.length);
            digits.emplace_back(ns);
            digits_freq[id] = other.digits_freq.at(i);
            mergeSegValues(digits[id], s);
        }
        else
        {
            digits_freq[id] += other.digits_freq.at(i);
            mergeSegValues(digits[id], s);
        }
    }

    // ---- symbols ----
    for (int i = 0; i < (int)other.symbols.size(); i++)
    {
        const segment &s = other.symbols[i];
        int id = FindSymbol(s);
        if (id == -1)
        {
            id = GetNextSymbolsID();
            segment ns(s.type, s.length);
            symbols.emplace_back(ns);
            symbols_freq[id] = other.symbols_freq.at(i);
            mergeSegValues(symbols[id], s);
        }
        else
        {
            symbols_freq[id] += other.symbols_freq.at(i);
            mergeSegValues(symbols[id], s);
        }
    }

    // ---- preterminals ----
    for (int i = 0; i < (int)other.preterminals.size(); i++)
    {
        const PT &pt      = other.preterminals[i];
        int       src_freq = other.preterm_freq.at(i);
        int       id       = FindPT(pt);
        if (id == -1)
        {
            id = GetNextPretermID();
            PT np = pt;
            if (np.curr_indices.empty())
                for (int j = 0; j < (int)np.content.size(); j++)
                    np.curr_indices.emplace_back(0);
            preterminals.emplace_back(np);
            preterm_freq[id] = src_freq;
        }
        else
        {
            preterm_freq[id] += src_freq;
        }
    }
}

void model::train(string path)
{
    // Phase 0: 单线程读文件
    cout << "Training..." << endl;
    cout << "Training phase 0: loading passwords into memory..." << endl;
    vector<string> passwords;
    passwords.reserve(3000000);
    {
        ifstream train_set(path);
        string pw;
        int lines = 0;
        while (train_set >> pw)
        {
            lines++;
            passwords.emplace_back(pw);
            if (lines % 10000 == 0)
                cout << "Lines loaded: " << lines << endl;
            if (lines >= 3000000)
                break;
        }
    }
    int total = (int)passwords.size();
    cout << "Total passwords loaded: " << total << endl;

    // Phase 1: 并行 parse，每个线程独立的 local_model
    cout << "Training phase 1: parallel parsing..." << endl;
    int nthreads = omp_get_max_threads();
    vector<model> local_models(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid   = omp_get_thread_num();
        int nthrs = omp_get_num_threads();
        int chunk = total / nthrs;
        int start = tid * chunk;
        int end   = (tid == nthrs - 1) ? total : start + chunk;
        for (int i = start; i < end; i++)
            local_models[tid].parse(passwords[i]);
    }

    // Phase 2: 串行合并
    cout << "Training phase 2: merging local models..." << endl;
    for (int t = 0; t < nthreads; t++)
        mergeFrom(local_models[t]);

    cout << "Merge complete. Total PTs: " << preterminals.size() << endl;
}

// ================================================================
// 以下所有函数与原版完全相同
// ================================================================

int model::FindPT(PT pt)
{
    for (int id = 0; id < (int)preterminals.size(); id++)
    {
        if (preterminals[id].content.size() != pt.content.size())
            continue;
        bool equal_flag = true;
        for (int idx = 0; idx < (int)preterminals[id].content.size(); idx++)
        {
            if (preterminals[id].content[idx].type   != pt.content[idx].type ||
                preterminals[id].content[idx].length != pt.content[idx].length)
            { equal_flag = false; break; }
        }
        if (equal_flag) return id;
    }
    return -1;
}

int model::FindLetter(segment seg)
{
    for (int id = 0; id < (int)letters.size(); id++)
        if (letters[id].length == seg.length) return id;
    return -1;
}

int model::FindDigit(segment seg)
{
    for (int id = 0; id < (int)digits.size(); id++)
        if (digits[id].length == seg.length) return id;
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < (int)symbols.size(); id++)
        if (symbols[id].length == seg.length) return id;
    return -1;
}

void PT::insert(segment seg) { content.emplace_back(seg); }

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
        freqs[values[value]] += 1;
}

void segment::order()
{
    for (pair<string, int> value : values)
        ordered_values.emplace_back(value.first);

    std::sort(ordered_values.begin(), ordered_values.end(),
              [this](const std::string &a, const std::string &b)
              { return freqs.at(values[a]) > freqs.at(values[b]); });

    // 原代码此处有两个完全相同的循环，保持原样以确保与原版行为一致
    for (const std::string &val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
    for (string val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0;

    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    { int id=GetNextDigitsID(); digits.emplace_back(seg); digits[id].insert(curr_part); digits_freq[id]=1; }
                    else
                    { int id=FindDigit(seg); digits_freq[id]+=1; digits[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    { int id=GetNextSymbolsID(); symbols.emplace_back(seg); symbols_freq[id]=1; symbols[id].insert(curr_part); }
                    else
                    { int id=FindSymbol(seg); symbols_freq[id]+=1; symbols[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
            }
            curr_type = 1; curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    { int id=GetNextLettersID(); letters.emplace_back(seg); letters_freq[id]=1; letters[id].insert(curr_part); }
                    else
                    { int id=FindLetter(seg); letters_freq[id]+=1; letters[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    { int id=GetNextSymbolsID(); symbols.emplace_back(seg); symbols_freq[id]=1; symbols[id].insert(curr_part); }
                    else
                    { int id=FindSymbol(seg); symbols_freq[id]+=1; symbols[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
            }
            curr_type = 2; curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    { int id=GetNextLettersID(); letters.emplace_back(seg); letters_freq[id]=1; letters[id].insert(curr_part); }
                    else
                    { int id=FindLetter(seg); letters_freq[id]+=1; letters[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    { int id=GetNextDigitsID(); digits.emplace_back(seg); digits_freq[id]=1; digits[id].insert(curr_part); }
                    else
                    { int id=FindDigit(seg); digits_freq[id]+=1; digits[id].insert(curr_part); }
                    curr_part.clear(); pt.insert(seg);
                }
            }
            curr_type = 3; curr_part += ch;
        }
    }

    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            { int id=GetNextLettersID(); letters.emplace_back(seg); letters_freq[id]=1; letters[id].insert(curr_part); }
            else
            { int id=FindLetter(seg); letters_freq[id]+=1; letters[id].insert(curr_part); }
            curr_part.clear(); pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            { int id=GetNextDigitsID(); digits.emplace_back(seg); digits_freq[id]=1; digits[id].insert(curr_part); }
            else
            { int id=FindDigit(seg); digits_freq[id]+=1; digits[id].insert(curr_part); }
            curr_part.clear(); pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            { int id=GetNextSymbolsID(); symbols.emplace_back(seg); symbols_freq[id]=1; symbols[id].insert(curr_part); }
            else
            { int id=FindSymbol(seg); symbols_freq[id]+=1; symbols[id].insert(curr_part); }
            curr_part.clear(); pt.insert(seg);
        }
    }

    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < (int)pt.content.size(); i++)
            pt.curr_indices.emplace_back(0);
        int id = GetNextPretermID();
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1) cout << "L" << length;
    if (type == 2) cout << "D" << length;
    if (type == 3) cout << "S" << length;
}

void segment::PrintValues()
{
    for (string iter : ordered_values)
        cout << iter << " freq:" << freqs[values[iter]] << endl;
}

void PT::PrintPT()
{
    for (auto iter : content) iter.PrintSeg();
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < (int)preterminals.size(); i++)
    {
        preterminals[i].PrintPT();
        cout << " freq:" << preterm_freq[i] << endl;
    }
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)] << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < (int)letters.size(); i++)
    { letters[i].PrintSeg(); cout << " freq:" << letters_freq[i] << endl; }
    for (int i = 0; i < (int)digits.size(); i++)
    { digits[i].PrintSeg(); cout << " freq:" << digits_freq[i] << endl; }
    for (int i = 0; i < (int)symbols.size(); i++)
    { symbols[i].PrintSeg(); cout << " freq:" << symbols_freq[i] << endl; }
}

bool compareByPretermProb(const PT &a, const PT &b)
{ return a.preterm_prob > b.preterm_prob; }

void model::order()
{
    cout << "Training phase 3: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    cout << "total pts: " << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    for (int i = 0; i < (int)letters.size(); i++) letters[i].order();
    cout << "Ordering digits" << endl;
    for (int i = 0; i < (int)digits.size(); i++) digits[i].order();
    cout << "Ordering symbols" << endl;
    for (int i = 0; i < (int)symbols.size(); i++) symbols[i].order();
}