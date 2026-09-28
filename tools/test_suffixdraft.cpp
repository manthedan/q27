// CPU unit tests for SuffixDraft (mirrors the toolconstrain.h test pattern).
#include <cstdio>
#include <vector>

#include "suffixdraft.h"

static int fails = 0;
#define CHECK(cond, name)                                                      \
    do {                                                                       \
        if (!(cond)) { printf("FAIL %s\n", name); fails++; }                   \
    } while (0)

int main() {
    using q27::SuffixDraft;
    // 1. basic recurrence: suffix (1,2,3,4) recurs; propose what followed
    {
        SuffixDraft s;
        s.reset({1, 2, 3, 4, 9, 8, 1, 2, 3});
        int out[4];
        int m = s.propose_with(4, 4, out); // conceptual stream ...1,2,3,[4]
        CHECK(m == 4, "basic match len");
        CHECK(out[0] == 9 && out[1] == 8 && out[2] == 1 && out[3] == 2, "basic continuation");
    }
    // 2. lag-0 exclusion: no earlier occurrence -> no match, even though the
    // suffix trivially matches itself
    {
        SuffixDraft s;
        s.reset({5, 6, 7, 8, 9, 10, 11, 12});
        int out[4];
        CHECK(s.propose_with(13, 4, out) == 0, "no self match");
    }
    // 3. periodic extension: run of identical tokens proposes more of them
    {
        SuffixDraft s;
        s.reset({7, 7, 7, 7});
        int out[6];
        int m = s.propose_with(7, 6, out); // conceptual 7,7,7,7,[7]; match p=3, lag 1
        CHECK(m == 4, "periodic match len");
        bool all7 = true;
        for (int i = 0; i < 6; i++) all7 &= out[i] == 7;
        CHECK(all7, "periodic extension");
    }
    // 4. most-recent occurrence wins ties: two occurrences of (1,2), longer
    // context differs; equal match length -> later position's continuation
    {
        SuffixDraft s;
        s.reset({9, 1, 2, 3, 4, 50, 60, 8, 1, 2, 3, 4, 70, 80, 5, 1, 2, 3});
        int out[2];
        int m = s.propose_with(4, 2, out); // suffix ...1,2,3,[4]: matches at p=4 and p=11
        CHECK(m >= 4, "tie match len");
        CHECK(out[0] == 70 && out[1] == 80, "most recent wins");
    }
    // 5. longest match beats recency: recent short match vs older longer one
    {
        SuffixDraft s;
        //          0   1  2  3  4  5   6   7  8  9  10 11 12  13
        s.reset({100, 20, 1, 2, 3, 4, 111, 30, 40, 1, 2, 3, 88, 20, 1, 2, 3});
        int out[1];
        // conceptual suffix: 20,1,2,3,[4] -- p=5 matches len>=5 (…20,1,2,3,4? t[5]=4:
        // key (2,3,4)... verify by API: match length must exceed 4 via the p=5 path
        int m = s.propose_with(4, 1, out);
        CHECK(m >= 5, "longest wins len");
        CHECK(out[0] == 111, "longest wins continuation");
    }
    // 6. append grows the index: match against generated tokens
    {
        SuffixDraft s;
        s.reset({1, 2, 3});
        for (int tok : {42, 43, 44, 45, 9, 42, 43, 44}) s.append(tok);
        int out[2];
        int m = s.propose_with(45, 2, out); // suffix 42,43,44,[45] recurs at gen pos
        CHECK(m == 4, "append match len");
        CHECK(out[0] == 9 && out[1] == 42, "append continuation");
    }
    // sync() builds exactly reset()'s index: random multi-turn continuations
    // over small vocabs (forces MAXCAND evictions), plus re-render drift
    {
        unsigned r = 12345;
        auto rnd = [&]() { r = r * 1103515245u + 12345u; return (int)((r >> 16) & 0x7fff); };
        int bad = 0, kept = 0;
        for (int trial = 0; trial < 200; trial++) {
            const int vocab = trial % 3 == 0 ? 3 : trial % 3 == 1 ? 20 : 5000;
            std::vector<int> p;
            SuffixDraft inc;
            for (int turn = 0; turn < 6; turn++) {
                if (turn && rnd() % 4 == 0 && p.size() > 50) p.resize(p.size() - rnd() % 40);
                const int add = 20 + rnd() % 400;
                for (int j = 0; j < add; j++) p.push_back(rnd() % vocab);
                kept += inc.sync(p) > 0;
                SuffixDraft ref;
                ref.reset(p);
                if (inc.index_digest() != ref.index_digest()) bad++;
                int o1[8], o2[8];
                for (int q = 0; q < 8; q++) {
                    const int pend = rnd() % vocab;
                    const int a = inc.propose_with(pend, 8, o1), b = ref.propose_with(pend, 8, o2);
                    if (a != b || (a && std::vector<int>(o1, o1 + 8) != std::vector<int>(o2, o2 + 8))) bad++;
                }
                for (int j = 0; j < 30; j++) inc.append(rnd() % vocab); // emitted tokens, then next turn
            }
        }
        CHECK(bad == 0, "sync == reset (index digest + proposals)");
        CHECK(kept > 100, "sync keeps the common prefix on continuations");
        printf("  sync vs reset: %d mismatches, %d incremental syncs of 1200\n", bad, kept);
    }
    if (fails == 0) printf("suffixdraft: all tests PASS\n");
    return fails ? 1 : 0;
}
