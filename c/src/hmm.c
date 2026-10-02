#include "hmm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline int32_t ln2_q(int F) {
  return (int32_t)(0.6931471805599453 * (float)(1 << F));
}

static int64_t exp_fp(int64_t x, int F) {
  if (x < -((int64_t)30 << F))
    return 0;
  if (x > ((int64_t)30 << F))
    return INT64_MAX / 4;

  const int32_t LN2_Q = ln2_q(F);

  // Range reduction: x = kxln2 + r, r e [-ln2/2, ln2/2]
  int64_t k = 0;
  int64_t r = x;
  while (r > LN2_Q / 2) {
    r -= LN2_Q;
    k++;
  }
  while (r < -LN2_Q / 2) {
    r += LN2_Q;
    k--;
  }

  // exp(r) por Taylor (|r| < 0.35, converge rápido)
  int64_t term = 1LL << F;
  int64_t sum = term;
  for (int i = 1; i < 10; i++) {
    term = (((int64_t)term * (int64_t)r) >> F) / i;
    sum += term;
    if (term == 0)
      break;
  }

  // sum ~ exp(r). Multiplica por 2^k.
  if (k >= 0) {
    if (k > 40)
      return INT64_MAX / 4;
    sum <<= k;
  } else {
    if (-k > 62)
      return 0;
    sum >>= -k;
  }
  return sum;
}

int32_t _log2_fp(int32_t x, int F) {
  if (x <= 0)
    return INT32_MIN;

  int32_t result = 0;
  int int_part = 0;

  const int64_t ONE = (1LL << F);
  const int64_t TWO = (2LL << F);

  if (x >= ONE) {
    while (x >= TWO) {
      x >>= 1;
      int_part++;
    }
  } else {
    while (x < ONE) {
      x <<= 1;
      int_part--;
    }
  }

  result = (int32_t)((int64_t)int_part * ONE);

  for (int i = 1; i <= F; i++) {
    x = (int32_t)(((int64_t)x * (int64_t)x) >> F);

    if (x >= TWO) {
      x >>= 1;
      result |= (int32_t)(1LL << (F - i));
    }
  }

  return result;
}

static inline int32_t log2pi_q(int F) {
  return (int32_t)(1.8378770664093453 * (float)(1 << F));
}

static int32_t logsumexp_fp(const int32_t *a, int n, int F) {
  int32_t m = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] > m)
      m = a[i];

  if (m == INT32_MIN)
    return INT32_MIN;

  int32_t sum = 0;
  for (int i = 0; i < n; i++) {
    int32_t d = a[i] - m;
    sum += exp_fp(d, F);
  }

  int32_t log2_s = (int32_t)_log2_fp(sum, F);
  int32_t ln_s = (int32_t)(((int64_t)log2_s * (int64_t)ln2_q(F)) >> F);

  int32_t res = (int32_t)m + ln_s;
  return (int32_t)res;
}

// Log-Gaussiana diagonal
static int32_t log_gaussian_fp(const int32_t *x, const int32_t *mu,
                               const int32_t *var, int D, int F) {
  const int32_t LN2_Q = ln2_q(F);
  const int32_t LOG2PI_Q = log2pi_q(F);

  int32_t maha = 0;
  int32_t log_det = 0;

  for (int d = 0; d < D; d++) {
    int32_t diff = x[d] - mu[d];

    //(x-μ)^2 em Q(F)
    int32_t diff_sq = (int32_t)(((int64_t)diff * (int64_t)diff) >> F);

    int32_t vv = var[d];
    if (vv <= 0)
      vv = 1;

    //(x-μ)^2/var em Q(F)
    int32_t quotient = (int32_t)((int64_t)diff_sq << F) / vv;
    maha += quotient;

    // ln(var) = log2(var) x ln(2)
    int32_t log2_v = (int32_t)_log2_fp(vv, F);
    log_det += (int32_t)((int64_t)log2_v * (int64_t)LN2_Q) >> F;
  }

  int32_t bracket = (D * LOG2PI_Q) + log_det + maha;
  int32_t res = -bracket / 2;
  return res;
}

// log_B[t][i] = log P(frame t | estado i)
static int32_t *log_emission_fp(const HMM_fp *m, int32_t **X, int T) {
  int n = (int)m->n_states;
  int mix = (int)m->n_mix;
  int D = (int)m->n_features;
  int F = (int)m->F;

  int32_t *log_B = malloc((size_t)T * n * sizeof(int32_t));
  int32_t *lg = malloc((size_t)mix * sizeof(int32_t));
  int32_t *log_w = malloc((size_t)mix * sizeof(int32_t));

  if (!log_B || !lg || !log_w) {
    free(log_B);
    free(lg);
    free(log_w);
    return NULL;
  }

  const int32_t LN2_Q = ln2_q(F);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < mix; k++) {
      int32_t l2w = (int32_t)_log2_fp(m->weights[i * mix + k], F);
      int32_t lnw = (int32_t)(((int64_t)l2w * (int64_t)LN2_Q) >> F);
      log_w[k] = lnw;
    }

    for (int t = 0; t < T; t++) {
      for (int k = 0; k < mix; k++) {
        const int32_t *mu = &m->means[(i * mix + k) * D];
        const int32_t *var = &m->covars[(i * mix + k) * D];
        int32_t lgf = log_gaussian_fp(X[t], mu, var, D, F);

        lg[k] = log_w[k] + lgf;
      }
      log_B[t * n + i] = logsumexp_fp(lg, mix, F);
    }
  }

  free(lg);
  free(log_w);
  return log_B;
}

// Apartir daqui decode.py / evaluate.py / hmm.py
int32_t hmm_score_fp(const HMM_fp *model, int32_t **X, int num_frames,
                     int num_ceps) {
  if (!model || !X || num_frames <= 0)
    return NAN;
  if (num_ceps != (int)model->n_features) {
    fprintf(stderr, "hmm_score_fp: dim mismatch: %d vs %u\n", num_ceps,
            model->n_features);
    return NAN;
  }

  int T = num_frames;
  int n = (int)model->n_states;
  int F = (int)model->F;

  int32_t *log_B = log_emission_fp(model, X, T);
  if (!log_B)
    return NAN;

  const int64_t LN2_Q = ln2_q(F);

  int32_t *log_self = malloc(n * sizeof(int32_t));
  int32_t *log_next = malloc(n * sizeof(int32_t));
  int32_t *v = malloc(n * sizeof(int32_t));
  int32_t *v_new = malloc(n * sizeof(int32_t));

  if (!log_self || !log_next || !v || !v_new) {
    free(log_B);
    free(log_self);
    free(log_next);
    free(v);
    free(v_new);
    return NAN;
  }

  for (int i = 0; i < n; i++) {
    int64_t l2_self = _log2_fp(model->transmat[i * n + i], F);
    log_self[i] = (int32_t)((int64_t)l2_self * (int64_t)LN2_Q) >> F;

    if (i < n - 1) {
      int64_t l2_next = _log2_fp(model->transmat[i * n + i + 1], F);
      log_next[i] = ((int64_t)l2_next * (int64_t)LN2_Q) >> F;
    } else {
      log_next[i] = INT32_MIN;
    }
  }

  for (int i = 0; i < n; i++)
    v[i] = INT32_MIN;
  v[0] = log_B[0 * n + 0];

  for (int t = 1; t < T; t++) {
    int64_t r0 =
        (int64_t)v[0] + (int64_t)log_self[0] + (int64_t)log_B[t * n + 0];
    if (r0 > INT32_MAX)
      r0 = INT32_MAX;
    if (r0 < INT32_MIN)
      r0 = INT32_MIN;
    v_new[0] = (int32_t)r0;

    for (int j = 1; j < n; j++) {
      int64_t stay = (int64_t)v[j] + (int64_t)log_self[j];
      int64_t advan = (int64_t)v[j - 1] + (int64_t)log_next[j - 1];
      int64_t best = (stay > advan) ? stay : advan;
      int64_t res = best + (int64_t)log_B[t * n + j];
      if (res > INT32_MAX)
        res = INT32_MAX;
      if (res < INT32_MIN)
        res = INT32_MIN;
      v_new[j] = (int32_t)res;
    }
    memcpy(v, v_new, n * sizeof(int32_t));
  }

  int32_t best = v[0];
  for (int i = 1; i < n; i++)
    if (v[i] > best)
      best = v[i];

  free(log_B);
  free(log_self);
  free(log_next);
  free(v);
  free(v_new);
  printf("%f\n", (float)best / (1 << F));
  return best;
}

// Viterbi geral — retorna caminho de estados
int32_t *hmm_decode_fp(const HMM_fp *model, int32_t **X, int num_frames,
                       int *out_T) {
  if (!model || !X || num_frames <= 0) {
    *out_T = 0;
    return NULL;
  }

  int T = num_frames;
  int n = (int)model->n_states;
  int F = (int)model->F;

  int32_t *log_B = log_emission_fp(model, X, T);
  if (!log_B) {
    *out_T = 0;
    return NULL;
  }

  int32_t *delta = malloc((size_t)T * n * sizeof(int32_t));
  int *psi = malloc((size_t)T * n * sizeof(int));
  int *path = malloc((size_t)T * sizeof(int));
  if (!delta || !psi || !path) {
    free(log_B);
    free(delta);
    free(psi);
    free(path);
    *out_T = 0;
    return NULL;
  }

  int32_t *log_A = malloc((size_t)n * n * sizeof(int32_t));
  int32_t *log_pi = malloc((size_t)n * sizeof(int32_t));
  if (!log_A || !log_pi) {
    free(log_B);
    free(delta);
    free(psi);
    free(path);
    free(log_A);
    free(log_pi);
    *out_T = 0;
    return NULL;
  }

  const int64_t LN2_Q = ln2_q(F);

  // log_pi[i] = ln(startprob[i]) em Q(F)
  for (int i = 0; i < n; i++) {
    int32_t pi = model->startprob[i];
    if (pi <= 0)
      pi = 1;
    int64_t l2 = _log2_fp(pi, F);
    int64_t ln = ((int64_t)l2 * (int64_t)LN2_Q) >> F;
    log_pi[i] = (int32_t)ln;
  }

  // log_A[i*n+j] = ln(transmat[i*n+j]) em Q(F)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int32_t a = model->transmat[i * n + j];
      if (a <= 0)
        a = 1;
      int64_t l2 = _log2_fp(a, F);
      int64_t ln = ((int64_t)l2 * (int64_t)LN2_Q) >> F;
      log_A[i * n + j] = (int32_t)ln;
    }
  }

  // Inicialização em t = 0
  for (int i = 0; i < n; i++) {
    int64_t s = (int64_t)log_pi[i] + (int64_t)log_B[0 * n + i];
    if (s > INT32_MAX)
      s = INT32_MAX;
    if (s < INT32_MIN)
      s = INT32_MIN;
    delta[0 * n + i] = (int32_t)s;
    psi[0 * n + i] = 0;
  }

  // Recursão t = 1..T-1
  for (int t = 1; t < T; t++) {
    for (int j = 0; j < n; j++) {
      int32_t best = INT32_MIN;
      int arg = 0;

      for (int i = 0; i < n; i++) {
        int64_t val =
            (int64_t)delta[(t - 1) * n + i] + (int64_t)log_A[i * n + j];
        if (val > INT32_MAX)
          val = INT32_MAX;
        if (val < INT32_MIN)
          val = INT32_MIN;
        int32_t v = (int32_t)val;
        if (v > best) {
          best = v;
          arg = i;
        }
      }

      psi[t * n + j] = arg;

      int64_t d = (int64_t)best + (int64_t)log_B[t * n + j];
      if (d > INT32_MAX)
        d = INT32_MAX;
      if (d < INT32_MIN)
        d = INT32_MIN;
      delta[t * n + j] = (int32_t)d;
    }
  }

  // Backtracking
  int32_t best = delta[(T - 1) * n + 0];
  int arg = 0;
  for (int i = 1; i < n; i++) {
    int32_t v = delta[(T - 1) * n + i];
    if (v > best) {
      best = v;
      arg = i;
    }
  }
  path[T - 1] = arg;

  for (int t = T - 2; t >= 0; t--)
    path[t] = psi[(t + 1) * n + path[t + 1]];

  free(log_B);
  free(delta);
  free(psi);
  free(log_A);
  free(log_pi);

  *out_T = T;
  return path;
}