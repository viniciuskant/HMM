#include "hmm_json.h"
#include "cJSON.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_whole_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return NULL;
  }
  fseek(f, 0, SEEK_SET);

  char *buf = malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[rd] = '\0';
  if (out_len)
    *out_len = rd;
  return buf;
}

static cJSON *get_field(cJSON *obj, const char *name) {
  return cJSON_GetObjectItemCaseSensitive(obj, name);
}

static int get_u32(cJSON *obj, const char *name, uint32_t *out) {
  cJSON *it = get_field(obj, name);
  if (!cJSON_IsNumber(it) || it->valuedouble < 0)
    return -1;
  *out = (uint32_t)it->valuedouble;
  return 0;
}

static int get_i32(cJSON *obj, const char *name, int *out) {
  cJSON *it = get_field(obj, name);
  if (!cJSON_IsNumber(it))
    return -1;
  *out = (int)it->valuedouble;
  return 0;
}

// Array 1D -> float*
static int read_farray_1d(cJSON *arr, float *dst, size_t expected) {
  if (!cJSON_IsArray(arr))
    return -1;
  if ((size_t)cJSON_GetArraySize(arr) != expected)
    return -1;
  size_t idx = 0;
  cJSON *it = NULL;
  cJSON_ArrayForEach(it, arr) {
    if (!cJSON_IsNumber(it))
      return -1;
    dst[idx++] = (float)it->valuedouble;
  }
  return 0;
}

// Array 2D (d1 × d2) -> float* row-major
static int read_farray_2d(cJSON *arr, float *dst, uint32_t d1, uint32_t d2) {
  if (!cJSON_IsArray(arr) || (uint32_t)cJSON_GetArraySize(arr) != d1)
    return -1;
  size_t idx = 0;
  for (uint32_t i = 0; i < d1; i++) {
    cJSON *row = cJSON_GetArrayItem(arr, (int)i);
    if (!cJSON_IsArray(row) || (uint32_t)cJSON_GetArraySize(row) != d2)
      return -1;
    for (uint32_t j = 0; j < d2; j++) {
      cJSON *v = cJSON_GetArrayItem(row, (int)j);
      if (!cJSON_IsNumber(v))
        return -1;
      dst[idx++] = (float)v->valuedouble;
    }
  }
  return 0;
}

// Array 3D (d1 × d2 × d3) -> float* row-major
static int read_farray_3d(cJSON *arr, float *dst, uint32_t d1, uint32_t d2,
                          uint32_t d3) {
  if (!cJSON_IsArray(arr) || (uint32_t)cJSON_GetArraySize(arr) != d1)
    return -1;
  size_t idx = 0;
  for (uint32_t i = 0; i < d1; i++) {
    cJSON *a2 = cJSON_GetArrayItem(arr, (int)i);
    if (!cJSON_IsArray(a2) || (uint32_t)cJSON_GetArraySize(a2) != d2)
      return -1;
    for (uint32_t j = 0; j < d2; j++) {
      cJSON *a3 = cJSON_GetArrayItem(a2, (int)j);
      if (!cJSON_IsArray(a3) || (uint32_t)cJSON_GetArraySize(a3) != d3)
        return -1;
      for (uint32_t k = 0; k < d3; k++) {
        cJSON *v = cJSON_GetArrayItem(a3, (int)k);
        if (!cJSON_IsNumber(v))
          return -1;
        dst[idx++] = (float)v->valuedouble;
      }
    }
  }
  return 0;
}

// carregar uma HMM

static void hmm_init(HMM *h) { memset(h, 0, sizeof(*h)); }

static void hmm_destroy(HMM *h) {
  free(h->startprob);
  free(h->transmat);
  free(h->weights);
  free(h->means);
  free(h->covars);
  hmm_init(h);
}

static void hmm_init_fp(HMM_fp *h) { memset(h, 0, sizeof(*h)); }
static void hmm_destroy_fp(HMM_fp *h) {
  free(h->startprob);
  free(h->transmat);
  free(h->weights);
  free(h->means);
  free(h->covars);
  hmm_init_fp(h);
}

static int load_hmm(cJSON *obj, HMM *h, const char *name) {
  hmm_init(h);

  if (!cJSON_IsObject(obj))
    return -1;

  if (get_u32(obj, "n_states", &h->n_states) != 0)
    return -1;
  if (get_u32(obj, "n_mix", &h->n_mix) != 0)
    return -1;
  if (get_u32(obj, "n_features", &h->n_features) != 0)
    return -1;

  size_t n = h->n_states;
  size_t m = h->n_mix;
  size_t D = h->n_features;

  h->startprob = malloc(n * sizeof(float));
  h->transmat = malloc(n * n * sizeof(float));
  h->weights = malloc(n * m * sizeof(float));
  h->means = malloc(n * m * D * sizeof(float));
  h->covars = malloc(n * m * D * sizeof(float));

  read_farray_1d(get_field(obj, "startprob"), h->startprob, n);
  read_farray_2d(get_field(obj, "transmat"), h->transmat, h->n_states,
                 h->n_states);
  read_farray_2d(get_field(obj, "weights"), h->weights, h->n_states, h->n_mix);
  read_farray_3d(get_field(obj, "means"), h->means, h->n_states, h->n_mix,
                 h->n_features);
  read_farray_3d(get_field(obj, "covars"), h->covars, h->n_states, h->n_mix,
                 h->n_features);

  return 0;
}

void copy_hmm_fp(const HMM *hmm, HMM_fp *hmm_fp, uint32_t F) {
  const int32_t SCALE = (int32_t)(1u << F);

  const size_t n = hmm->n_states;
  const size_t m = hmm->n_mix;
  const size_t D = hmm->n_features;

  hmm_fp->F = (uint32_t)F;
  hmm_fp->n_states = (uint32_t)n;
  hmm_fp->n_mix = (uint32_t)m;
  hmm_fp->n_features = (uint32_t)D;

  hmm_fp->startprob = malloc(n * sizeof(int32_t));
  hmm_fp->transmat = malloc(n * n * sizeof(int32_t));
  hmm_fp->weights = malloc(n * m * sizeof(int32_t));
  hmm_fp->means = malloc(n * m * D * sizeof(int32_t));
  hmm_fp->covars = malloc(n * m * D * sizeof(int32_t));

  for (size_t i = 0; i < n; i++)
    hmm_fp->startprob[i] = (int32_t)(hmm->startprob[i] * SCALE);

  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      hmm_fp->transmat[i * n + j] = (int32_t)(hmm->transmat[i * n + j] * SCALE);

  for (size_t i = 0; i < n; i++)
    for (size_t k = 0; k < m; k++)
      hmm_fp->weights[i * m + k] = (int32_t)(hmm->weights[i * m + k] * SCALE);

  for (size_t i = 0; i < n; i++)
    for (size_t k = 0; k < m; k++)
      for (size_t d = 0; d < D; d++)
        hmm_fp->means[(i * m + k) * D + d] =
            (int32_t)(hmm->means[(i * m + k) * D + d] * SCALE);

  for (size_t i = 0; i < n; i++)
    for (size_t k = 0; k < m; k++)
      for (size_t d = 0; d < D; d++)
        hmm_fp->covars[(i * m + k) * D + d] =
            (int32_t)(hmm->covars[(i * m + k) * D + d] * SCALE);
}

void hmm_bundle_free(HMMBundle *b) {
  if (!b)
    return;
  hmm_destroy(&b->word);
  hmm_destroy(&b->filler);
  hmm_destroy(&b->silence);
  b->has_silence = 0;
  b->dim = 0;
}

int hmm_bundle_load_json(const char *path, HMMBundle_fp *out, uint32_t F) {
  HMMBundle b;
  memset(&b, 0, sizeof(b));
  memset(out, 0, sizeof(*out));

  size_t len = 0;
  char *text = read_whole_file(path, &len);
  if (!text) {
    fprintf(stderr, "não abriu %s\n", path);
    return -1;
  }

  cJSON *root = cJSON_ParseWithLength(text, len);
  free(text);
  if (!root) {
    const char *err = cJSON_GetErrorPtr();
    fprintf(stderr, "JSON inválido perto de: %.40s\n", err ? err : "?");
    return -1;
  }

  int rc = -1;
  if (get_i32(root, "dim", &b.dim) != 0) {
    fprintf(stderr, "campo 'dim' ausente\n");
    cJSON_Delete(root);
    hmm_bundle_free(&b);
    return rc;
  }

  out->dim = b.dim;

  if (load_hmm(get_field(root, "word"), &b.word, "word") != 0) {
    cJSON_Delete(root);
    hmm_bundle_free(&b);
    return rc;
  }
  copy_hmm_fp(&b.word, &out->word, F);

  if (load_hmm(get_field(root, "filler"), &b.filler, "filler") != 0) {
    cJSON_Delete(root);
    hmm_bundle_free(&b);
    return rc;
  }
  copy_hmm_fp(&b.filler, &out->filler, F);

  cJSON *sil = get_field(root, "silence");
  if (sil && !cJSON_IsNull(sil)) {
    if (load_hmm(sil, &b.silence, "silence") != 0) {
      cJSON_Delete(root);
      hmm_bundle_free(&b);
      return rc;
    }
    copy_hmm_fp(&b.silence, &out->silence, F);
    b.has_silence = 1;
    out->has_silence = 1;
  } else {
    b.has_silence = 0;
    out->has_silence = 0;
  }

  rc = 0;
  cJSON_Delete(root);
  hmm_bundle_free(&b);
  return rc;
}

void hmm_bundle_free_fp(HMMBundle_fp *b) {
  if (!b)
    return;
  hmm_destroy_fp(&b->word);
  hmm_destroy_fp(&b->filler);
  hmm_destroy_fp(&b->silence);
  b->has_silence = 0;
  b->dim = 0;
}

void hmm_bundle_dump_fp(const HMMBundle_fp *b) {
  printf("dim = %d\n", b->dim);

  const char *names[] = {"word", "filler", "silence"};

  const HMM_fp *items[] = {&b->word, &b->filler,
                           b->has_silence ? &b->silence : NULL};

  for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
    const HMM_fp *h = items[i];

    if (!h) {
      printf("  %-8s (ausente)\n", names[i]);
      continue;
    }

    size_t n = h->n_states;
    size_t m = h->n_mix;
    size_t D = h->n_features;

    printf("  %-8s  n_states=%zu  n_mix=%zu  n_features=%zu\n", names[i], n, m,
           D);

    printf("    startprob = [");
    for (size_t a = 0; a < n; a++)
      printf("%d%s", h->startprob[a], a + 1 < n ? ", " : "");
    printf("]\n");

    printf("    transmat  (linha 0): [");
    for (size_t j = 0; j < n; j++)
      printf("%d%s", h->transmat[j], j + 1 < n ? ", " : "");
    printf("]\n");

    printf("    means[0][0][0..4] = [");
    for (size_t d = 0; d < D && d < 5; d++)
      printf("%d%s", h->means[(0 * m + 0) * D + d],
             d + 1 < D && d + 1 < 5 ? ", " : "");
    printf("]\n");

    printf("    covars[0][0][0..4] = [");
    for (size_t d = 0; d < D && d < 5; d++)
      printf("%d%s", h->covars[(0 * m + 0) * D + d],
             d + 1 < D && d + 1 < 5 ? ", " : "");
    printf("]\n");
  }
}