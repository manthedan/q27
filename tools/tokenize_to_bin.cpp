// Tokenize a raw text file into the flat uint32 token stream that
// q27-metal --nll / CUDA --nll consume. Usage:
//   tokenize_to_bin TOKENIZER.tok INPUT.txt OUTPUT.bin
#include "tokenizer.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TOKENIZER.tok INPUT.txt OUTPUT.bin\n", argv[0]);
        return 1;
    }
    q27::Tokenizer tokenizer(argv[1]);
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::stringstream ss; ss << in.rdbuf();
    std::vector<int> ids = tokenizer.encode(ss.str());
    std::vector<uint32_t> out(ids.begin(), ids.end());
    FILE* f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    if (fwrite(out.data(), 4, out.size(), f) != out.size()) {
        fprintf(stderr, "short write\n"); return 1;
    }
    fclose(f);
    printf("%zu tokens -> %s\n", out.size(), argv[3]);
    return 0;
}
