//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "R.h"
#include "Rmath.h"


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size2 = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word2 {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[1024], output_file_in[1024], output_file_out[1024];
char save_vocab_file2[MAX_STRING], read_vocab_file2[MAX_STRING];
struct vocab_word2 *vocab2;
int binary2 = 0, cbow2 = 0, debug_mode2 = 2, window2 = 12, min_count2 = 5, num_threads2 = 1, min_reduce2 = 1;
int *vocab_hash2;
long long vocab_max_size2 = 1000, vocab_size2 = 0, layer1_size2 = 100;
long long train_words2 = 0, word_count_actual2 = 0, iter2 = 5, file_size2 = 0;
real alpha2 = 0.025, starting_alpha2, sample2 = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start2;

int hs2 = 1, negative2 = 0;
const int table_size2 = 1e8;
int *table;


void InitUnigramTable2() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size2 * sizeof(int));
  for (a = 0; a < vocab_size2; a++) train_words_pow += pow(vocab2[a].cn, power);
  i = 0;
  d1 = pow(vocab2[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size2; a++) {
    table[a] = i;
    if (a / (real)table_size2 > d1) {
      i++;
      d1 += pow(vocab2[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size2) i = vocab_size2 - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord2(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash2(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size2;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab2(char *word) {
  unsigned int hash = GetWordHash2(word);
  while (1) {
    if (vocab_hash2[hash] == -1) return -1;
    if (!strcmp(word, vocab2[vocab_hash2[hash]].word)) return vocab_hash2[hash];
    hash = (hash + 1) % vocab_hash_size2;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex2(FILE *fin) {
  char word[MAX_STRING];
  ReadWord2(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab2(word);
}

// Adds a word to the vocabulary
int AddWordToVocab2(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab2[vocab_size2].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab2[vocab_size2].word, word);
  vocab2[vocab_size2].cn = 0;
  vocab_size2++;
  // Reallocate memory if needed
  if (vocab_size2 + 2 >= vocab_max_size2) {
    vocab_max_size2 += 1000;
    vocab2 = (struct vocab_word2 *)realloc(vocab2, vocab_max_size2 * sizeof(struct vocab_word2));
  }
  hash = GetWordHash2(word);
  while (vocab_hash2[hash] != -1) hash = (hash + 1) % vocab_hash_size2;
  vocab_hash2[hash] = vocab_size2 - 1;
  return vocab_size2 - 1;
}

// Used later for sorting by word counts
int VocabCompare2(const void *a, const void *b) {
    return ((struct vocab_word2 *)b)->cn - ((struct vocab_word2 *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab2() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab2[1], vocab_size2 - 1, sizeof(struct vocab_word2), VocabCompare2);
  for (a = 0; a < vocab_hash_size2; a++) vocab_hash2[a] = -1;
  size = vocab_size2;
  train_words2 = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab2[a].cn < min_count2) {
      vocab_size2--;
      free(vocab2[vocab_size2].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash2(vocab2[a].word);
      while (vocab_hash2[hash] != -1) hash = (hash + 1) % vocab_hash_size2;
      vocab_hash2[hash] = a;
      train_words2 += vocab2[a].cn;
    }
  }
  vocab2 = (struct vocab_word2 *)realloc(vocab2, (vocab_size2 + 1) * sizeof(struct vocab_word2));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size2; a++) {
    vocab2[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab2[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab2() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size2; a++) if (vocab2[a].cn > min_reduce2) {
    vocab2[b].cn = vocab2[a].cn;
    vocab2[b].word = vocab2[a].word;
    b++;
  } else free(vocab2[a].word);
  vocab_size2 = b;
  for (a = 0; a < vocab_hash_size2; a++) vocab_hash2[a] = -1;
  for (a = 0; a < vocab_size2; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash2(vocab2[a].word);
    while (vocab_hash2[hash] != -1) hash = (hash + 1) % vocab_hash_size2;
    vocab_hash2[hash] = a;
  }
  fflush(NULL);
  min_reduce2++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree2() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size2 * 2 + 1, sizeof(long long));
  long long *binary2 = (long long *)calloc(vocab_size2 * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size2 * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size2; a++) count[a] = vocab2[a].cn;
  for (a = vocab_size2; a < vocab_size2 * 2; a++) count[a] = 1e15;
  pos1 = vocab_size2 - 1;
  pos2 = vocab_size2;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size2 - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size2 + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size2 + a;
    parent_node[min2i] = vocab_size2 + a;
    binary2[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size2; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary2[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size2 * 2 - 2) break;
    }
    vocab2[a].codelen = i;
    vocab2[a].point[0] = vocab_size2 - 2;
    for (b = 0; b < i; b++) {
      vocab2[a].code[i - b - 1] = code[b];
      vocab2[a].point[i - b] = point[b] - vocab_size2;
    }
  }
  free(count);
  free(binary2);
  free(parent_node);
}

void LearnVocabFromTrainFile2() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size2; a++) vocab_hash2[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    Rprintf("ERROR: training data file not found!\n");
    Rf_error("Error!");
  }
  vocab_size2 = 0;
  AddWordToVocab2((char *)"</s>");
  while (1) {
    ReadWord2(word, fin);
    if (feof(fin)) break;
    train_words2++;
        if ((debug_mode2 > 1) && (train_words2 % 100000 == 0)) {
          Rprintf("%lldK%c", train_words2 / 1000, 13);
          fflush(NULL);
    }
    i = SearchVocab2(word);
    if (i == -1) {
      a = AddWordToVocab2(word);
      vocab2[a].cn = 1;
    } else vocab2[i].cn++;
    if (vocab_size2 > vocab_hash_size2 * 0.7) ReduceVocab2();
  }
  SortVocab2();
  if (debug_mode2 > 0) {
    Rprintf("Vocab size: %lld\n", vocab_size2);
    Rprintf("Words in train file: %lld\n", train_words2);
  }
  file_size2 = ftell(fin);
  fclose(fin);
}

void SaveVocab2() {
  long long i;
  FILE *fo = fopen(save_vocab_file2, "wb");
  for (i = 0; i < vocab_size2; i++) fprintf(fo, "%s %lld\n", vocab2[i].word, vocab2[i].cn);
  fclose(fo);
}

void ReadVocab2() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file2, "rb");
  if (fin == NULL) {
    Rprintf("Vocabulary file not found\n");
    Rf_error("Error!");
  }
  for (a = 0; a < vocab_hash_size2; a++) vocab_hash2[a] = -1;
  vocab_size2 = 0;
  while (1) {
    ReadWord2(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab2(word);
    if(fscanf(fin, "%lld%c", &vocab2[a].cn, &c)==1)
      ;
    i++;
  }
  SortVocab2();
  if (debug_mode2 > 0) {
    Rprintf("Vocab size: %lld\n", vocab_size2);
    Rprintf("Words in train file: %lld\n", train_words2);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    Rprintf("ERROR: training data file not found!\n");
    Rf_error("Error!");
  }
  fseek(fin, 0, SEEK_END);
  file_size2 = ftell(fin);
  fclose(fin);
}

void InitNet2() {
  long long a, b;
  unsigned long long next_random = 1;
//  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#ifdef _WIN32
  syn0 = (real *)_aligned_malloc((long long)vocab_size2 * layer1_size2 * sizeof(real), 128);
#else
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#endif

  if (syn0 == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
  if (hs2) {
//    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#ifdef _WIN32
    syn1 = (real *)_aligned_malloc((long long)vocab_size2 * layer1_size2 * sizeof(real), 128);
#else
    a = posix_memalign((void **)&(syn1), 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#endif

   if (syn1 == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    for (a = 0; a < vocab_size2; a++) for (b = 0; b < layer1_size2; b++)
     syn1[a * layer1_size2 + b] = 0;
  }
  if (negative2>0) {
   // a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#ifdef _WIN32
    syn1neg = (real *)_aligned_malloc((long long)vocab_size2 * layer1_size2 * sizeof(real), 128);
#else
    a = posix_memalign((void **)&(syn1neg), 128, (long long)vocab_size2 * layer1_size2 * sizeof(real));
#endif

if (syn1neg == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    for (a = 0; a < vocab_size2; a++) for (b = 0; b < layer1_size2; b++)
     syn1neg[a * layer1_size2 + b] = 0;
  }
  for (a = 0; a < vocab_size2; a++) for (b = 0; b < layer1_size2; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size2 + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size2;
  }
  CreateBinaryTree2();
}

void *TrainModelThread2(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter2;
  unsigned long long next_random = (long long)id;
  //  real doneness_f, speed_f;
  // For writing to R.
  real f, g;
  real *neu1 = (real *)calloc(layer1_size2, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size2, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size2 / (long long)num_threads2 * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual2 += word_count - last_word_count;
      last_word_count = word_count;
      /* if ((debug_mode2 > 1)) { */
      /*   now=clock(); */
      /* 	doneness_f = word_count_actual2 / (real)(iter2 * train_words2 + 1) * 100; */
      /* 	speed_f = word_count_actual2 / ((real)(now - start2 + 1) / (real)CLOCKS_PER_SEC * 1000); */

      /*   Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, */
      /* 		doneness_f, speed_f); */

      /*   fflush(NULL); */
      /* } */
      alpha2 = starting_alpha2 * (1 - word_count_actual2 / (real)(iter2 * train_words2 + 1));
      if (alpha2 < starting_alpha2 * 0.0001) alpha2 = starting_alpha2 * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex2(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample2 > 0) {
          real ran = (sqrt(vocab2[word].cn / (sample2 * train_words2)) + 1) * (sample2 * train_words2) / vocab2[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words2 / num_threads2)) {
       word_count_actual2 += word_count - last_word_count;
       local_iter--;
       if (local_iter == 0) break;
       word_count = 0;
       last_word_count = 0;
       sentence_length = 0;
       fseek(fi, file_size2 / (long long)num_threads2 * (long long)id, SEEK_SET);
       continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size2; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size2; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window2;
    if (cbow2) {  //train the cbow2 architecture
      // in -> hidden
      for (a = b; a < window2 * 2 + 1 - b; a++) if (a != window2) {
        c = sentence_position - window2 + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size2; c++) neu1[c] += syn0[c + last_word * layer1_size2];
      }
      if (hs2) for (d = 0; d < vocab2[word].codelen; d++) {
        f = 0;
        l2 = vocab2[word].point[d] * layer1_size2;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size2; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab2[word].code[d] - f) * alpha2;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size2; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size2; c++) syn1[c + l2] += g * neu1[c];
      }
      // NEGATIVE SAMPLING
      if (negative2 > 0) for (d = 0; d < negative2 + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size2];
          if (target == 0) target = next_random % (vocab_size2 - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size2;
        f = 0;
        for (c = 0; c < layer1_size2; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha2;
        else if (f < -MAX_EXP) g = (label - 0) * alpha2;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha2;
        for (c = 0; c < layer1_size2; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size2; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window2 * 2 + 1 - b; a++) if (a != window2) {
        c = sentence_position - window2 + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size2; c++) syn0[c + last_word * layer1_size2] += neu1e[c];
      }
    } else {  //train skip-gram
      for (a = b; a < window2 * 2 + 1 - b; a++) if (a != window2) {
        c = sentence_position - window2 + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size2;
        for (c = 0; c < layer1_size2; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs2) for (d = 0; d < vocab2[word].codelen; d++) {
          f = 0;
          l2 = vocab2[word].point[d] * layer1_size2;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size2; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab2[word].code[d] - f) * alpha2;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size2; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size2; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative2 > 0) for (d = 0; d < negative2 + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size2];
            if (target == 0) target = next_random % (vocab_size2 - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size2;
          f = 0;
          for (c = 0; c < layer1_size2; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha2;
          else if (f < -MAX_EXP) g = (label - 0) * alpha2;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha2;
          for (c = 0; c < layer1_size2; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size2; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size2; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel2() {
  long a, b;
  FILE *fo_in, *fo_out;
  pthread_t *pt = (pthread_t *)malloc(num_threads2 * sizeof(pthread_t));
  Rprintf("Starting training using file %s\n", train_file);
  starting_alpha2 = alpha2;
  if (read_vocab_file2[0] != 0) ReadVocab2(); else LearnVocabFromTrainFile2();
  if (save_vocab_file2[0] != 0) SaveVocab2();
  if (output_file_in[0] == 0) return;
  if (output_file_out[0] == 0) return;
  InitNet2();
  if (negative2 > 0) InitUnigramTable2();
  start2 = clock();
  for (a = 0; a < num_threads2; a++) pthread_create(&pt[a], NULL, TrainModelThread2, (void *)a);
  for (a = 0; a < num_threads2; a++) pthread_join(pt[a], NULL);

  fo_in = fopen(output_file_in, "wb");

  fprintf(fo_in, "%lld %lld\n", vocab_size2, layer1_size2);
    for (a = 0; a < vocab_size2; a++) {
      fprintf(fo_in, "%s ", vocab2[a].word);
      if (binary2) for (b = 0; b < layer1_size2; b++) fwrite(&syn0[a * layer1_size2 + b], sizeof(real), 1, fo_in);
      else for (b = 0; b < layer1_size2; b++) fprintf(fo_in, "%lf ", syn0[a * layer1_size2 + b]);
      fprintf(fo_in, "\n");
  }

  fclose(fo_in);

  fo_out = fopen(output_file_out, "wb");

  if(negative2 == 0){
      fprintf(fo_out, "%lld %lld\n", vocab_size2, layer1_size2);
      for (a = 0; a < vocab_size2; a++) {
        fprintf(fo_out, "%s ", vocab2[a].word);
        if (binary2) for (b = 0; b < layer1_size2; b++) fwrite(&syn1[a * layer1_size2 + b], sizeof(real), 1, fo_out);
        else for (b = 0; b < layer1_size2; b++) fprintf(fo_out, "%lf ", syn1[a * layer1_size2 + b]);
        fprintf(fo_out, "\n");
      }
  }else{
       fprintf(fo_out, "%lld %lld\n", vocab_size2, layer1_size2);
        for (a = 0; a < vocab_size2; a++) {
        fprintf(fo_out, "%s ", vocab2[a].word);
        if (binary2) for (b = 0; b < layer1_size2; b++) fwrite(&syn1neg[a * layer1_size2 + b], sizeof(real), 1, fo_out);
        else for (b = 0; b < layer1_size2; b++) fprintf(fo_out, "%lf ", syn1neg[a * layer1_size2 + b]);
        fprintf(fo_out, "\n");
      }
  }

  fclose(fo_out);
}

int ArgPos2(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      Rprintf("Argument missing for %s\n", str);
      Rf_error("Error!");
    }
    return a;
  }
  return -1;
}

