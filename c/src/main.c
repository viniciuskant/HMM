#include "hmm_json.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "hmm_bundle.json";
    uint32_t F = 16;

    HMMBundle_fp b_fp;
    if (hmm_bundle_load_json(path, &b_fp, F) != 0) {
        fprintf(stderr, "falha carregando %s\n", path);
        return 1;
    }

    hmm_bundle_dump_fp(&b_fp);

    hmm_bundle_free_fp(&b_fp);
    return 0;
}