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

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[1024], output_file_in[1024], output_file_out[1024], output_file_in_left[1024], output_file_in_right[1024], output_file_out_left[1024], output_file_out_right[1024];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 12, min_count = 5, num_threads = 1, min_reduce = 1;
int two_in = 0, two_out = 0;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0l, *syn0r, *syn0, *syn1l, *syn1r, *syn1, *syn1lneg, *syn1rneg, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;


void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
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
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(NULL);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
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
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    Rprintf("ERROR: training data file not found!\n");
    Rf_error("Error!");
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
          Rprintf("%lldK%c", train_words / 1000, 13);
          fflush(NULL);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    Rprintf("Vocab size: %lld\n", vocab_size);
    Rprintf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    Rprintf("Vocabulary file not found\n");
    Rf_error("Error!");
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    if(fscanf(fin, "%lld%c", &vocab[a].cn, &c)==1)
      ;
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    Rprintf("Vocab size: %lld\n", vocab_size);
    Rprintf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    Rprintf("ERROR: training data file not found!\n");
    Rf_error("Error!");
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long al, ar, a, b;
  unsigned long long next_random = 1;
#ifdef _WIN32

  if(two_in == 1){
    syn0l = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
    syn0r = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }else{
    syn0 = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }
  
#else

  if(two_in == 1){
    al = posix_memalign((void **)&syn0l, 128, (long long)vocab_size * layer1_size * sizeof(real));
    ar = posix_memalign((void **)&syn0r, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }else{
    alr = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }
  
#endif

  if(two_in == 1){
    if (syn0l == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    if (syn0r == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
  } else{
    if (syn0 == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
  }
  
if (hs) {
//    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef _WIN32

  if(two_out == 1){
    syn1l = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
    syn1r = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }else{
    syn1 = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }

#else

  if(two_out == 1){
    al = posix_memalign((void **)&syn1l, 128, (long long)vocab_size * layer1_size * sizeof(real));
    ar = posix_memalign((void **)&syn1r, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }else{
    alr = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }

 //   a = posix_memalign((void **)&(syn1), 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

  if(two_out == 1){
    if (syn1l == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    if (syn1r == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1l[a * layer1_size + b] = 0;

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1r[a * layer1_size + b] = 0;
  } else{
    if (syn1 == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;

  }

 }
  if (negative>0) {
   // a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef _WIN32

  if(two_out == 1){
    syn1lneg = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
    syn1rneg = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }else{
    syn1neg = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  }

#else

  if(two_out == 1){
    al = posix_memalign((void **)&syn1lneg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    ar = posix_memalign((void **)&syn1rneg, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }else{
    alr = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
  }

#endif


if(two_out == 1){
    if (syn1lneg == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    if (syn1rneg == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1lneg[a * layer1_size + b] = 0;

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1rneg[a * layer1_size + b] = 0;
  } else{
    if (syn1neg == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;

  }

  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    if(two_in == 1){
      syn0l[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      syn0r[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }else{
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
  }
 /* for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0r[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  } */
  CreateBinaryTree();
}

void *TrainModelThread_bi(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long) id ;
  //  real doneness_f, speed_f;
  // For writing to R.
  real f, g;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      /* if ((debug_mode > 1)) { */
      /*   now=clock(); */
      /* 	doneness_f = word_count_actual / (real)(iter * train_words + 1) * 100; */
      /* 	speed_f = word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000); */

      /*   Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, */
      /* 		doneness_f, speed_f); */

      /*   fflush(NULL); */
      /* } */
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
       word_count_actual += word_count - last_word_count;
       local_iter--;
       if (local_iter == 0) break;
       word_count = 0;
       last_word_count = 0;
       sentence_length = 0;
       fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
       continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (long int cc = 0; cc < layer1_size; cc++) neu1[cc] = 0;
    for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      // For the words occurring in the left window

      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        if(c < sentence_position && c > 0){
          for (long int cc = 0; cc < layer1_size; cc++) neu1[cc] += syn0l[cc + last_word * layer1_size];
        }else{
          for (long int cc = 0; cc < layer1_size; cc++) neu1[cc] += syn0r[cc + last_word * layer1_size];
        }
      }

      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1[cc + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (long int cc= 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1[cc + l2];
        // Learn weights hidden -> output
        for (long int cc = 0; cc < layer1_size; cc++) syn1[cc + l2] += g * neu1[cc];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1neg[cc + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1neg[cc + l2];
        for (long int cc = 0; cc < layer1_size; cc++) syn1neg[cc + l2] += g * neu1[cc];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        if(c < sentence_position && c > 0){
          for (long int cc = 0; cc < layer1_size; cc++) syn0l[cc + last_word * layer1_size] += neu1e[cc];
        }else{
          for (long int cc = 0; cc < layer1_size; cc++) syn0r[cc + last_word * layer1_size] += neu1e[cc];
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          if(c < sentence_position && c > 0){
            for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1[cc + l2];
          }else{
            for (long int cc = 0; cc < layer1_size; cc++) f += syn0r[cc + l1] * syn1[cc + l2];
          }
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1[cc + l2];
          // Learn weights hidden -> output
          if(c < sentence_position && c > 0){
            for (long int cc = 0; cc < layer1_size; cc++) syn1[cc + l2] += g * syn0l[cc + l1];
          }else{
            for (long int cc = 0; cc < layer1_size; cc++) syn1[cc + l2] += g * syn0r[cc + l1];
          }
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          if(c < sentence_position){
            for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1neg[cc + l2];
          }else{
            for (long int cc = 0; cc < layer1_size; cc++) f += syn0r[cc + l1] * syn1neg[cc + l2];
          }
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1neg[cc + l2];
          if(c < sentence_position && c > 0){
            for (long int cc = 0; cc < layer1_size; cc++) syn1neg[cc + l2] += g * syn0l[cc + l1];
          }else{
            for (long int cc = 0; cc < layer1_size; cc++) syn1neg[cc + l2] += g * syn0r[cc + l1];
          }
        }
        // Learn weights input -> hidden
        if(c < sentence_position && c > 0){
          for (long int cc = 0; cc < layer1_size; cc++) syn0l[cc + l1] += neu1e[cc];
        }else{
          for (long int cc = 0; cc < layer1_size; cc++) syn0r[cc + l1] += neu1e[cc];
        }
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


void *TrainModelThread_left(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long) id ;
  //  real doneness_f, speed_f;
  // For writing to R.
  real f = 0, g = 0;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      /* if ((debug_mode > 1)) { */
      /*   now=clock(); */
      /*  doneness_f = word_count_actual / (real)(iter * train_words + 1) * 100; */
      /*  speed_f = word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000); */

      /*   Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, */
      /*    doneness_f, speed_f); */

      /*   fflush(NULL); */
      /* } */
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
       word_count_actual += word_count - last_word_count;
       local_iter--;
       if (local_iter == 0) break;
       word_count = 0;
       last_word_count = 0;
       sentence_length = 0;
       fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
       continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (long int cc = 0; cc < layer1_size; cc++) neu1[cc] = 0;
    for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if(b > window ) continue;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      // For the words occurring in the left window
          for (a = b; a < window ; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (long int cc = 0; cc < layer1_size; cc++) neu1[cc] += syn0l[cc + last_word * layer1_size];
          }

          if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1l[cc + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
          for (long int cc= 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1l[cc + l2];
        // Learn weights hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) syn1l[cc + l2] += g * neu1[cc];
        } 
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
          target = word;
          label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
        }
          l2 = target * layer1_size;
          f = 0;
          for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1lneg[cc + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1lneg[cc + l2];
          for (long int cc = 0; cc < layer1_size; cc++) syn1lneg[cc + l2] += g * neu1[cc];
        }
        // hidden -> in
        for (a = b; a < window ; a++) if (a != window) {
           c = sentence_position - window + a;
           if (c < 0) continue;
           if (c >= sentence_length) continue;
           last_word = sen[c];
           if (last_word == -1) continue;
           for (long int cc = 0; cc < layer1_size; cc++) syn0l[cc + last_word * layer1_size] += neu1e[cc];
        }
    } else {  //train skip-gram
      for (a = b; a < window ; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1l[cc + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1l[cc + l2];
          // Learn weights hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) syn1l[cc + l2] += g * syn0l[cc + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1lneg[cc + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1lneg[cc + l2];
          for (long int cc = 0; cc < layer1_size; cc++) syn1lneg[cc + l2] += g * syn0l[cc + l1];
        }
        // Learn weights input -> hidden
      }
    }
      
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1l[cc + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (long int cc= 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1l[cc + l2];
        // Learn weights hidden -> output
        for (long int cc = 0; cc < layer1_size; cc++) syn1l[cc + l2] += g * neu1[cc];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (long int cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1lneg[cc + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1lneg[cc + l2];
        for (long int cc = 0; cc < layer1_size; cc++) syn1lneg[cc + l2] += g * neu1[cc];
      }
      // hidden -> in
      for (a = b; a < window ; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (long int cc = 0; cc < layer1_size; cc++) syn0l[cc + last_word * layer1_size] += neu1e[cc];
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1l[cc + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1l[cc + l2];
          // Learn weights hidden -> output
          for (long int cc = 0; cc < layer1_size; cc++) syn1l[cc + l2] += g * syn0l[cc + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (long int cc = 0; cc < layer1_size; cc++) f += syn0l[cc + l1] * syn1lneg[cc + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (long int cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1lneg[cc + l2];
          for (long int cc = 0; cc < layer1_size; cc++) syn1lneg[cc + l2] += g * syn0l[cc + l1];
        }
        // Learn weights input -> hidden
          for (long int cc = 0; cc < layer1_size; cc++) syn0l[cc + l1] += neu1e[cc];
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



void *TrainModelThread_right(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  //  real doneness_f, speed_f;
  // For writing to R.
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;      
      /* if ((debug_mode > 1)) { */
      /*   now=clock(); */
      /*  doneness_f = word_count_actual / (real)(iter * train_words + 1) * 100; */
      /*  speed_f = word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000); */
    
      /*   Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, */
      /*    doneness_f, speed_f); */

      /*   fflush(NULL); */
      /* } */
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
       word_count_actual += word_count - last_word_count;
       local_iter--;
       if (local_iter == 0) break;
       word_count = 0;
       last_word_count = 0;
       sentence_length = 0;
       fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
       continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if((window*2 + 1 - b) < (window+1)) continue;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      for (a = (window+1); a < window * 2 + 1 - b; a++) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (cc = 0; cc < layer1_size; cc++) neu1[cc] += syn0r[cc + last_word * layer1_size];
      }
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1r[cc + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1r[cc + l2];
        // Learn weights hidden -> output
        for (cc = 0; cc < layer1_size; cc++) syn1r[cc + l2] += g * neu1[cc];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (cc = 0; cc < layer1_size; cc++) f += neu1[cc] * syn1rneg[cc + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1rneg[cc + l2];
        for (cc = 0; cc < layer1_size; cc++) syn1rneg[cc + l2] += g * neu1[cc];
      }
      // hidden -> in
      for (a = (window+1); a < window * 2 + 1 - b; a++){
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (cc = 0; cc < layer1_size; cc++) syn0r[cc + last_word * layer1_size] += neu1e[cc];
      }
    } else {  //train skip-gram
      for (a = (window+1); a < window * 2 + 1 - b; a++) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (cc = 0; cc < layer1_size; cc++) neu1e[cc] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (cc = 0; cc < layer1_size; cc++) f += syn0r[cc + l1] * syn1r[cc + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1r[cc + l2];
          // Learn weights hidden -> output
          for (cc = 0; cc < layer1_size; cc++) syn1r[cc + l2] += g * syn0r[cc + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (cc = 0; cc < layer1_size; cc++) f += syn0r[cc + l1] * syn1rneg[cc + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (cc = 0; cc < layer1_size; cc++) neu1e[cc] += g * syn1rneg[cc + l2];
          for (cc = 0; cc < layer1_size; cc++) syn1rneg[cc + l2] += g * syn0r[cc + l1];
        }
        // Learn weights input -> hidden
        for (cc = 0; cc < layer1_size; cc++) syn0r[cc + l1] += neu1e[cc];
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


void *TrainModelThread_uni(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  //  real doneness_f, speed_f;
  // For writing to R.
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;      
      /* if ((debug_mode > 1)) { */
      /*   now=clock(); */
      /*  doneness_f = word_count_actual / (real)(iter * train_words + 1) * 100; */
      /*  speed_f = word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000); */
    
      /*   Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, */
      /*    doneness_f, speed_f); */

      /*   fflush(NULL); */
      /* } */
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
       word_count_actual += word_count - last_word_count;
       local_iter--;
       if (local_iter == 0) break;
       word_count = 0;
       last_word_count = 0;
       sentence_length = 0;
       fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
       continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      }
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
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


void TrainModel() {
  long a, b;
  FILE *fo_left, *fo_right, *fo_out;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  Rprintf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file_left[0] == 0) return;
  if (output_file_right[0] == 0) return;
  if (output_file_out[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();

  if(two_in == 0 && two_out == 0){
             File *fo, *fi;
             if (output_file_in[0] == 0) return;
             if (output_file_out[0] == 0) return;
             for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread_uni, (void *)a);
             for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);


             Rprintf("Saving the input word vectors \n");

              fi = fopen(output_file_in, "wb");


             // Save the input word vectors

             fprintf(fi, "%lld %lld\n", vocab_size, layer1_size);
              for (a = 0; a < vocab_size; a++) {
                fprintf(fi, "%s ", vocab[a].word);
                if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fi);
                else for (b = 0; b < layer1_size; b++) fprintf(fi, "%lf ", syn0[a * layer1_size + b]);
                fprintf(fi, "\n");
              }

              fclose (fi);

              Rprintf("Saving the output word vectors \n");

              fo = fopen(output_file_out, "wb");


             // Save the output word vectors

              if(negative == 0){
                fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
                for (a = 0; a < vocab_size; a++) {
                fprintf(fo, "%s ", vocab[a].word);
                if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo);
                else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1[a * layer1_size + b]);
                fprintf(fo, "\n");
                }

                fclose (fo);
              }else{
                fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
                for (a = 0; a < vocab_size; a++) {
                fprintf(fo, "%s ", vocab[a].word);
                if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
                else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
                fprintf(fo, "\n");
                }

                fclose (fo);
              }

  } else if (two_in == 1 && two_out == 0){

               File *fo, *fi_left, *fi_right;
               if (output_file_in_left[0] == 0) return;
               if (output_file_in_right[0] == 0) return;
               if (output_file_out[0] == 0) return;
               for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread_bi, (void *)a);
               for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);


               Rprintf("Saving the left input word vectors \n");

                fi_left = fopen(output_file_in_left, "wb");

               fprintf(fi_left, "%lld %lld\n", vocab_size, layer1_size);
                for (a = 0; a < vocab_size; a++) {
                  fprintf(fi_left, "%s ", vocab[a].word);
                  if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0l[a * layer1_size + b], sizeof(real), 1, fi_left);
                  else for (b = 0; b < layer1_size; b++) fprintf(fi_left, "%lf ", syn0l[a * layer1_size + b]);
                  fprintf(fi_left, "\n");
                }

                fclose (fi_left);

                Rprintf("Saving the right input word vectors \n");

                fi_right = fopen(output_file_in_right, "wb");

               fprintf(fi_right, "%lld %lld\n", vocab_size, layer1_size);
                for (a = 0; a < vocab_size; a++) {
                  fprintf(fi_right, "%s ", vocab[a].word);
                  if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0r[a * layer1_size + b], sizeof(real), 1, fi_right);
                  else for (b = 0; b < layer1_size; b++) fprintf(fi_right, "%lf ", syn0r[a * layer1_size + b]);
                  fprintf(fi_right, "\n");
                }

                fclose (fi_right);

                Rprintf("Saving the output word vectors \n");


                if(negative == 0){
                   fo = fopen(output_file_out, "wb");

                   fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
                   for (a = 0; a < vocab_size; a++) {
                      fprintf(fo, "%s ", vocab[a].word);
                      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo);
                      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1[a * layer1_size + b]);
                      fprintf(fo, "\n");
                   }

                   fclose (fo);

                }else{
                   fo = fopen(output_file_out, "wb");

                   fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
                   for (a = 0; a < vocab_size; a++) {
                      fprintf(fo, "%s ", vocab[a].word);
                      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
                      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
                      fprintf(fo, "\n");
                   }

                   fclose (fo);
                }
      
  } else if (two_in == 1  && two_out == 1){

             File *fo_left, *fo_right, *fi_left, *fi_right;
             if (output_file_in_left[0] == 0) return;
             if (output_file_in_right[0] == 0) return;
             if (output_file_out_left[0] == 0) return;
             if (output_file_out_right[0] == 0) return;

             for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread_left, (void *)a);
             for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

             for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread_right, (void *)a);
             for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

             Rprintf("Saving the left input word vectors \n");

              fi_left = fopen(output_file_in_left, "wb");

             fprintf(fi_left, "%lld %lld\n", vocab_size, layer1_size);
              for (a = 0; a < vocab_size; a++) {
                fprintf(fi_left, "%s ", vocab[a].word);
                if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0l[a * layer1_size + b], sizeof(real), 1, fi_left);
                else for (b = 0; b < layer1_size; b++) fprintf(fi_left, "%lf ", syn0l[a * layer1_size + b]);
                fprintf(fi_left, "\n");
              }

              fclose (fi_left);

              Rprintf("Saving the right input word vectors \n");

              fi_right = fopen(output_file_in_right, "wb");

             fprintf(fi_right, "%lld %lld\n", vocab_size, layer1_size);
              for (a = 0; a < vocab_size; a++) {
                fprintf(fi_right, "%s ", vocab[a].word);
                if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0r[a * layer1_size + b], sizeof(real), 1, fi_right);
                else for (b = 0; b < layer1_size; b++) fprintf(fi_right, "%lf ", syn0r[a * layer1_size + b]);
                fprintf(fi_right, "\n");
              }

              fclose (fi_right);


      if(negative == 0){

            Rprintf("Saving the left output word vectors \n");

            fo_left = fopen(output_file_out_left, "wb");

           fprintf(fo_left, "%lld %lld\n", vocab_size, layer1_size);
            for (a = 0; a < vocab_size; a++) {
              fprintf(fo_left, "%s ", vocab[a].word);
              if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1l[a * layer1_size + b], sizeof(real), 1, fo_left);
              else for (b = 0; b < layer1_size; b++) fprintf(fo_left, "%lf ", syn1l[a * layer1_size + b]);
              fprintf(fo_left, "\n");
            }

            fclose (fo_left);

            Rprintf("Saving the right utput word vectors \n");

            fo_right = fopen(output_file_out_right, "wb");

            fprintf(fo_right, "%lld %lld\n", vocab_size, layer1_size);
            for (a = 0; a < vocab_size; a++) {
              fprintf(fo_right, "%s ", vocab[a].word);
              if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1r[a * layer1_size + b], sizeof(real), 1, fo_right);
              else for (b = 0; b < layer1_size; b++) fprintf(fo_right, "%lf ", syn1r[a * layer1_size + b]);
              fprintf(fo_right, "\n");
            }

            fclose (fo_right);

      } else{
              Rprintf("Saving the left output word vectors \n");

              fo_left = fopen(output_file_out_left, "wb");

              fprintf(fo_left, "%lld %lld\n", vocab_size, layer1_size);
              for (a = 0; a < vocab_size; a++) {
              fprintf(fo_left, "%s ", vocab[a].word);
              if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1lneg[a * layer1_size + b], sizeof(real), 1, fo_left);
              else for (b = 0; b < layer1_size; b++) fprintf(fo_left, "%lf ", syn1lneg[a * layer1_size + b]);
              fprintf(fo_left, "\n");
              }

              fclose (fo_left);

               Rprintf("Saving the right utput word vectors \n");

              fo_right = fopen(output_file_out_right, "wb");

              fprintf(fo_right, "%lld %lld\n", vocab_size, layer1_size);
              for (a = 0; a < vocab_size; a++) {
              fprintf(fo_right, "%s ", vocab[a].word);
              if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1rneg[a * layer1_size + b], sizeof(real), 1, fo_right);
              else for (b = 0; b < layer1_size; b++) fprintf(fo_right, "%lf ", syn1rneg[a * layer1_size + b]);
              fprintf(fo_right, "\n");
             }

            fclose (fo_right);
      }}
     else{
       Rf_error("Error! two_in cannot be true when two_out is false");
    }
}

int ArgPos(char *str, int argc, char **argv) {
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

