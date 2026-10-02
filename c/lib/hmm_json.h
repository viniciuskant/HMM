#ifndef HMM_JSON_H
#define HMM_JSON_H

#include <stddef.h>
#include <stdint.h>

/* Layout row-major, igual NumPy:
 *   transmat[i*n_states + j]
 *   weights [i*n_mix + k]
 *   means   [(i*n_mix + k)*n_features + d]
 *   covars  [(i*n_mix + k)*n_features + d]
 */
typedef struct {
  uint32_t n_states;
  uint32_t n_mix;
  uint32_t n_features;

  float *startprob; // [n_states]
  float *transmat;  // [n_states * n_states]
  float *weights;   // [n_states * n_mix]
  float *means;     // [n_states * n_mix * n_feat]
  float *covars;    // [n_states * n_mix * n_feat]
} HMM;

typedef struct {
  int dim; // dimensão do MFCC (bundle["dim"])
  HMM word;
  HMM filler;
  HMM silence;
  int has_silence; // 0 se "silence" era null no JSON
} HMMBundle;

typedef struct {
  uint32_t n_states;
  uint32_t n_mix;
  uint32_t n_features;
  uint32_t F;

  int32_t *startprob; // [n_states]
  int32_t *transmat;  // [n_states * n_states]
  int32_t *weights;   // [n_states * n_mix]
  int32_t *means;     // [n_states * n_mix * n_feat]
  int32_t *covars;    // [n_states * n_mix * n_feat]
} HMM_fp;

typedef struct {
  int dim; // n coeficientes do MFCC
  HMM_fp word;
  HMM_fp filler;
  HMM_fp silence;
  int has_silence; // 0 se "silence" era null no JSON
} HMMBundle_fp;

int hmm_bundle_load_json(const char *path, HMMBundle_fp *out, uint32_t F);

void hmm_bundle_dump_fp(const HMMBundle_fp *b);
void hmm_bundle_free_fp(HMMBundle_fp *b);

#endif