#include "hmm.h"
#include "mfcc/mfcc.h"
#include "wav.h"

#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double BPF_SOS[][6] = {
    {0.57173198331682984, 1.1434639666336597, 0.57173198331682984, 1,
     1.3625219619531816, 0.47604180545601055},
    {1, 2, 1, 1, 1.6124225251649913, 0.74552454077471841},
    {1, -2, 1, 1, -1.9423852416050995, 0.9433544885950067},
    {1, -2, 1, 1, -1.975371880085838, 0.97634885779931668},
};

#define BPF_N_SECTIONS ((int)(sizeof(BPF_SOS) / sizeof(BPF_SOS[0])))

#define BPF_PADLEN (3 * (2 * BPF_N_SECTIONS + 1))
#define CONFIG_FILE "config.txt"

static void sosfilt_forward(const double *x, int n, double *y) {
  memcpy(y, x, (size_t)n * sizeof(double));
  for (int s = 0; s < BPF_N_SECTIONS; s++) {
    const double b0 = BPF_SOS[s][0], b1 = BPF_SOS[s][1], b2 = BPF_SOS[s][2];
    const double a1 = BPF_SOS[s][4], a2 = BPF_SOS[s][5];
    double z1 = 0.0, z2 = 0.0;
    for (int i = 0; i < n; i++) {
      double in = y[i];
      double out = b0 * in + z1;
      z1 = b1 * in - a1 * out + z2;
      z2 = b2 * in - a2 * out;
      y[i] = out;
    }
  }
}

static int bandpass_int16(const int16_t *in, int n, int16_t *out) {
  if (n <= BPF_PADLEN + 1) {
    memcpy(out, in, (size_t)n * sizeof(int16_t));
    return 0;
  }

  const int pad = BPF_PADLEN;
  const int total = n + 2 * pad;

  double *ext = malloc((size_t)total * sizeof(double));
  double *tmp = malloc((size_t)total * sizeof(double));
  if (!ext || !tmp) {
    free(ext);
    free(tmp);
    return -1;
  }

  for (int i = 0; i < pad; i++)
    ext[pad - 1 - i] = 2.0 * in[0] - in[1 + i];
  for (int i = 0; i < n; i++)
    ext[pad + i] = (double)in[i];
  for (int i = 0; i < pad; i++)
    ext[pad + n + i] = 2.0 * in[n - 1] - in[n - 2 - i];

  sosfilt_forward(ext, total, tmp);

  for (int i = 0; i < total / 2; i++) {
    double t = tmp[i];
    tmp[i] = tmp[total - 1 - i];
    tmp[total - 1 - i] = t;
  }

  sosfilt_forward(tmp, total, ext);

  for (int i = 0; i < total / 2; i++) {
    double t = ext[i];
    ext[i] = ext[total - 1 - i];
    ext[total - 1 - i] = t;
  }

  for (int i = 0; i < n; i++) {
    double v = ext[pad + i];
    if (v > 32767.0)
      v = 32767.0;
    if (v < -32768.0)
      v = -32768.0;
    out[i] = (int16_t)lround(v);
  }

  free(ext);
  free(tmp);
  return 0;
}

static void trim_line(char *line) {
  char *comment = strchr(line, '#');
  if (comment)
    *comment = '\0';
  char *end = line + strlen(line) - 1;
  while (end >= line && isspace((unsigned char)*end))
    end--;
  *(end + 1) = '\0';
  char *start = line;
  while (*start && isspace((unsigned char)*start))
    start++;
  if (start != line)
    memmove(line, start, strlen(start) + 1);
}

int load_config(const char *dir, mfcc_config_t *cfg) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, CONFIG_FILE);
  FILE *f = fopen(path, "r");
  if (!f) {
    printf("Arquivo de configuração não encontrado em %s, usando defaults.\n",
           path);
    cfg->F_PRE = 12;
    cfg->F_HAMMING = 12;
    cfg->F_FFT = 12;
    cfg->F_MEL = 12;
    cfg->F_DCT = 12;
    cfg->TRUNCATE_PRE = 0;
    cfg->TRUNCATE_HAMMING = 0;
    cfg->TRUNCATE_FFT = 0;
    cfg->TRUNCATE_MEL = 0;
    cfg->TRUNCATE_DCT = 0;
    return 0;
  }

  char line[256];
  while (fgets(line, sizeof(line), f)) {
    trim_line(line);
    if (strlen(line) == 0)
      continue;

    char key[64], value[64];
    if (sscanf(line, "%63[^=]=%63s", key, value) != 2) {
      fprintf(stderr, "Linha ignorada (formato inválido): %s\n", line);
      continue;
    }

    int val = atoi(value);

    if (strcmp(key, "F_PRE") == 0)
      cfg->F_PRE = val;
    else if (strcmp(key, "F_HAMMING") == 0)
      cfg->F_HAMMING = val;
    else if (strcmp(key, "F_FFT") == 0)
      cfg->F_FFT = val;
    else if (strcmp(key, "F_MEL") == 0)
      cfg->F_MEL = val;
    else if (strcmp(key, "F_DCT") == 0)
      cfg->F_DCT = val;
    else if (strcmp(key, "TRUNCATE_PRE") == 0)
      cfg->TRUNCATE_PRE = val;
    else if (strcmp(key, "TRUNCATE_HAMMING") == 0)
      cfg->TRUNCATE_HAMMING = val;
    else if (strcmp(key, "TRUNCATE_FFT") == 0)
      cfg->TRUNCATE_FFT = val;
    else if (strcmp(key, "TRUNCATE_MEL") == 0)
      cfg->TRUNCATE_MEL = val;
    else if (strcmp(key, "TRUNCATE_DCT") == 0)
      cfg->TRUNCATE_DCT = val;
    else
      fprintf(stderr, "Chave desconhecida: %s\n", key);
  }
  fclose(f);
  return 1;
}

static void apply_cmvn(mfcc_result_t *r, int F) {
  int T = r->num_frames;
  int D = r->num_ceps;
  if (T <= 0 || D <= 0)
    return;

  const double scale = (double)(1LL << F);

  double *mean = calloc((size_t)D, sizeof(double));
  double *var = calloc((size_t)D, sizeof(double));
  if (!mean || !var) {
    free(mean);
    free(var);
    return;
  }

  for (int t = 0; t < T; t++)
    for (int j = 0; j < D; j++)
      mean[j] += (double)r->coefficients[t][j] / scale;

  for (int j = 0; j < D; j++)
    mean[j] /= (double)T;

  for (int t = 0; t < T; t++)
    for (int j = 0; j < D; j++) {
      double d = (double)r->coefficients[t][j] / scale - mean[j];
      var[j] += d * d;
    }

  for (int j = 0; j < D; j++)
    var[j] = sqrt(var[j] / (double)T);

  for (int t = 0; t < T; t++)
    for (int j = 0; j < D; j++) {
      double x = (double)r->coefficients[t][j] / scale;
      double y = (x - mean[j]) / (var[j] + 1e-8);

      int64_t q = (int64_t)llround(y * scale);
      if (q > INT32_MAX)
        q = INT32_MAX;
      if (q < INT32_MIN)
        q = INT32_MIN;
      r->coefficients[t][j] = (int32_t)q;
    }

  free(mean);
  free(var);
}

typedef struct {
  double word;
  double filler;
  double silence;
  int has_silence;
} score_triple_t;

static score_triple_t score_audio_triple(void *bundle_ptr, const char *filename,
                                         mfcc_config_t cfg) {
  score_triple_t out = {NAN, NAN, NAN, 0};
  HMMBundle_fp *bundle = (HMMBundle_fp *)bundle_ptr;

  int16_t *samples = NULL;
  uint32_t data_bytes = 0;
  WavHeader *header = open_wav_file(filename, &samples, &data_bytes);
  if (!header) {
    free(samples);
    return out;
  }

  int sample_rate = header->sampleRate;
  int num_samples = (int)(data_bytes / sizeof(int16_t));
  int16_t *samples_bp = malloc((size_t)num_samples * sizeof(int16_t));

  if (!samples_bp) {
    free(samples);
    free(header);
    return out;
  }
  if (bandpass_int16(samples, num_samples, samples_bp) != 0) {
    fprintf(stderr, "Bandpass filter failed: %s\n", filename);
    free(samples_bp);
    free(samples);
    free(header);
    return out;
  }

  mfcc_result_t result = {0};
  if (mfcc_compute(samples, num_samples, sample_rate, &cfg, &result) != 0) {
    free(samples);
    free(header);
    return out;
  }
  // TODO: colocar os deltas

  apply_cmvn(&result, cfg.F_DCT);

  double T = (double)result.num_frames;

  out.word = hmm_score_fp(&bundle->word, result.coefficients, result.num_frames,
                          result.num_ceps) /
             T;
  out.filler = hmm_score_fp(&bundle->filler, result.coefficients,
                            result.num_frames, result.num_ceps) /
               T;
  if (bundle->has_silence) {
    out.silence = hmm_score_fp(&bundle->silence, result.coefficients,
                               result.num_frames, result.num_ceps) /
                  T;
    out.has_silence = 1;
  }

  mfcc_free_result(&result);
  free(samples);
  free(samples_bp);
  free(header);
  return out;
}

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(const char **)a, *(const char **)b);
}

score_triple_t *collect_scores(void *bundle, const char *folder,
                               size_t *count) {
  *count = 0;

  DIR *dir = opendir(folder);
  if (!dir)
    return NULL;

  size_t cap = 16, n = 0;
  char **names = malloc(cap * sizeof(char *));
  struct dirent *entry;

  while ((entry = readdir(dir)) != NULL) {
    if (!strstr(entry->d_name, ".wav"))
      continue;
    if (n == cap) {
      cap *= 2;
      names = realloc(names, cap * sizeof(char *));
    }
    names[n++] = strdup(entry->d_name);
  }
  closedir(dir);

  qsort(names, n, sizeof(char *), cmp_str);
  char *c_dir = getenv("C_DIR");
  mfcc_config_t cfg;
  load_config(c_dir, &cfg);
  score_triple_t *scores = malloc(n * sizeof(score_triple_t));
  for (size_t i = 0; i < n; i++) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", folder, names[i]);
    scores[i] = score_audio_triple(bundle, path, cfg);
    free(names[i]);
  }
  free(names);

  *count = n;
  return scores;
}

static double delta_of(score_triple_t s) {
  double m = s.filler;
  if (s.has_silence && s.silence > m)
    m = s.silence;
  return s.word - m;
}

typedef struct {
  double P, R, F1;
  int TP, FP, FN, TN;
} metrics_t;

static metrics_t metrics_at(double threshold, const double *pos, size_t n_pos,
                            const double *neg, size_t n_neg) {
  metrics_t m = {0, 0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < n_pos; i++) {
    if (pos[i] > threshold)
      m.TP++;
    else
      m.FN++;
  }
  for (size_t i = 0; i < n_neg; i++) {
    if (neg[i] > threshold)
      m.FP++;
    else
      m.TN++;
  }

  m.P = (m.TP + m.FP) ? (double)m.TP / (m.TP + m.FP) : 0.0;
  m.R = (m.TP + m.FN) ? (double)m.TP / (m.TP + m.FN) : 0.0;
  m.F1 = (m.P + m.R) ? 2.0 * m.P * m.R / (m.P + m.R) : 0.0;
  return m;
}

double tune_threshold(void *bundle, const char *data_dir) {
  char pos_dir[1024], neg_dir[1024];
  snprintf(pos_dir, sizeof(pos_dir), "%s/verification/positive", data_dir);
  snprintf(neg_dir, sizeof(neg_dir), "%s/verification/negative", data_dir);

  size_t n_pos = 0, n_neg = 0;
  score_triple_t *sp = collect_scores(bundle, pos_dir, &n_pos);
  score_triple_t *sn = collect_scores(bundle, neg_dir, &n_neg);

  if (!n_pos || !n_neg) {
    fprintf(stderr, "Faltam dados em %s / %s\n", pos_dir, neg_dir);
    free(sp);
    free(sn);
    exit(1);
  }

  double *pos_d = malloc(n_pos * sizeof(double));
  double *neg_d = malloc(n_neg * sizeof(double));
  for (size_t i = 0; i < n_pos; i++)
    pos_d[i] = delta_of(sp[i]);
  for (size_t i = 0; i < n_neg; i++)
    neg_d[i] = delta_of(sn[i]);

  size_t n_all = n_pos + n_neg;
  double *all_d = malloc(n_all * sizeof(double));
  memcpy(all_d, pos_d, n_pos * sizeof(double));
  memcpy(all_d + n_pos, neg_d, n_neg * sizeof(double));

  double mn = all_d[0], mx = all_d[0];
  for (size_t i = 1; i < n_all; i++) {
    if (all_d[i] < mn)
      mn = all_d[i];
    if (all_d[i] > mx)
      mx = all_d[i];
  }

  const int GRID_N = 400;
  double best_f1 = -1.0, best_t = 0.0;
  double best_p = 0.0, best_r = 0.0;

  for (int i = 0; i < GRID_N; i++) {
    double t = mn + (mx - mn) * (double)i / (double)(GRID_N - 1);
    metrics_t m = metrics_at(t, pos_d, n_pos, neg_d, n_neg);
    if (m.F1 > best_f1) {
      best_f1 = m.F1;
      best_t = t;
      best_p = m.P;
      best_r = m.R;
    }
  }

  printf("[verification] melhor F1=%.3f em threshold=%.4f (P=%.3f, R=%.3f)\n",
         best_f1, best_t, best_p, best_r);

  free(all_d);
  free(pos_d);
  free(neg_d);
  free(sp);
  free(sn);
  return best_t;
}

void report_test(void *bundle, const char *data_dir, double threshold) {
  char pos_dir[1024], neg_dir[1024];
  snprintf(pos_dir, sizeof(pos_dir), "%s/test/positive", data_dir);
  snprintf(neg_dir, sizeof(neg_dir), "%s/test/negative", data_dir);

  size_t n_pos = 0, n_neg = 0;
  score_triple_t *sp = collect_scores(bundle, pos_dir, &n_pos);
  score_triple_t *sn = collect_scores(bundle, neg_dir, &n_neg);

  if (!n_pos || !n_neg) {
    fprintf(stderr, "Faltam dados em %s / %s\n", pos_dir, neg_dir);
    free(sp);
    free(sn);
    exit(1);
  }

  double *pos_d = malloc(n_pos * sizeof(double));
  double *neg_d = malloc(n_neg * sizeof(double));
  for (size_t i = 0; i < n_pos; i++)
    pos_d[i] = delta_of(sp[i]);
  for (size_t i = 0; i < n_neg; i++)
    neg_d[i] = delta_of(sn[i]);

  metrics_t m = metrics_at(threshold, pos_d, n_pos, neg_d, n_neg);

  printf(
      "[test] threshold=%.4f P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d TN=%d)\n",
      threshold, m.P, m.R, m.F1, m.TP, m.FP, m.FN, m.TN);

  free(pos_d);
  free(neg_d);
  free(sp);
  free(sn);
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "hmm_bundle.json";
  uint32_t F = 12;

  HMMBundle_fp b_fp;
  if (hmm_bundle_load_json(path, &b_fp, F) != 0) {
    fprintf(stderr, "falha carregando %s\n", path);
    return 1;
  }

  const char *data_dir = getenv("DATA_DIR");
  if (!data_dir) {
    fprintf(stderr, "Erro: DATA_DIR não definida\n");
    hmm_bundle_free_fp(&b_fp);
    return 1;
  }

  double t = tune_threshold(&b_fp, data_dir);
  report_test(&b_fp, data_dir, t);

  printf("\nUse THRESHOLD = %.4f no config.py\n", t);

  hmm_bundle_free_fp(&b_fp);
  return 0;
}