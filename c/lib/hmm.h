#ifndef HMM_H
#define HMM_H

#include <stddef.h>
#include <stdint.h>
#include "hmm_json.h"

int32_t hmm_score_fp(const HMM_fp *model, int32_t **X, int num_frames,
                    int num_ceps);

int32_t *hmm_decode_fp(const HMM_fp *model, int32_t **X, int num_frames,
                   int *out_T);

#endif