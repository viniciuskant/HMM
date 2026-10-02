#include "wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

WavHeader *open_wav_file(const char *filename, int16_t **samples,
                         uint32_t *out_data_size) {

  WavHeader *header = (WavHeader *)malloc(sizeof(WavHeader));
  if (!header) {
    fprintf(stderr, "Memory allocation failed for WavHeader\n");
    return NULL;
  }

  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    free(header);
    return NULL;
  }

  if (fread(header, sizeof(WavHeader), 1, file) != 1) {
    fprintf(stderr, "Falha lendo cabeçalho WAV: %s\n", filename);
    fclose(file);
    free(header);
    return NULL;
  }

  if (strncmp(header->chunkID, "RIFF", 4) != 0 ||
      strncmp(header->format, "WAVE", 4) != 0) {
    fprintf(stderr, "Invalid WAV file format: %s\n", filename);
    fclose(file);
    free(header);
    return NULL;
  }

  if (header->audioFormat != 1) {
    fprintf(stderr, "Formato de áudio não é PCM (linear): %s\n", filename);
    fclose(file);
    free(header);
    return NULL;
  }

  if (header->subchunk1Size < 16) {
    fprintf(stderr, "Subchunk1Size inválido (%u bytes): %s\n",
            header->subchunk1Size, filename);
    fclose(file);
    free(header);
    return NULL;
  }

  if (header->subchunk1Size > 16) {
    long skip = (long)header->subchunk1Size - 16;
    if (fseek(file, skip, SEEK_CUR) != 0) {
      fprintf(stderr, "Falha ao pular extensão do fmt: %s\n", filename);
      fclose(file);
      free(header);
      return NULL;
    }
  }

  char chunkID[4];
  uint32_t chunkSize;
  int found_data = 0;

  while (!found_data) {
    if (fread(chunkID, 1, 4, file) != 4 ||
        fread(&chunkSize, sizeof(uint32_t), 1, file) != 1) {
      fprintf(stderr, "Falha ao ler cabeçalho de chunk: %s\n", filename);
      fclose(file);
      free(header);
      return NULL;
    }

    if (strncmp(chunkID, "data", 4) == 0) {
      found_data = 1;
      break;
    }

    long skip = (long)chunkSize + (chunkSize & 1);
    if (fseek(file, skip, SEEK_CUR) != 0) {
      fprintf(stderr, "Falha ao pular chunk '%c%c%c%c': %s\n", chunkID[0],
              chunkID[1], chunkID[2], chunkID[3], filename);
      fclose(file);
      free(header);
      return NULL;
    }
  }

  int16_t *buf = (int16_t *)malloc(chunkSize);
  if (!buf) {
    fprintf(stderr, "Memory allocation failed for samples (%u bytes): %s\n",
            chunkSize, filename);
    fclose(file);
    free(header);
    return NULL;
  }

  if (fread(buf, 1, chunkSize, file) != chunkSize) {
    fprintf(stderr, "Leitura incompleta das amostras: %s\n", filename);
    free(buf);
    fclose(file);
    free(header);
    return NULL;
  }

  fclose(file);

  *samples = buf;
  *out_data_size = chunkSize;
  return header;
}