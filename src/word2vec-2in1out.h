#include <stdio.h>
#include "R.h"
#include "Rmath.h"
#include "tools.h"


void InitNet2to1() {
  long long al, ar, a, b;
  unsigned long long next_random = 1;
#ifdef _WIN32

  syn0l = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  syn0r = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#else
  al = posix_memalign((void **)&syn0l, 128, (long long)vocab_size * layer1_size * sizeof(real));
  ar = posix_memalign((void **)&syn0r, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

  if (syn0l == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
  if (syn0r == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}

if (hs) {
//    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef _WIN32
    syn1 = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#else
    a = posix_memalign((void **)&(syn1), 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

   if (syn1 == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
   // a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef _WIN32
    syn1neg = (real *)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#else
    a = posix_memalign((void **)&(syn1neg), 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

if (syn1neg == NULL) {Rprintf("Memory allocation failed\n"); Rf_error("Error!");}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0l[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    syn0r[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
 /* for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0r[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  } */
  CreateBinaryTree();
}

void *TrainModelThread2to1(void *id) {
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

void TrainModel2to1() {
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
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread2to1, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  fo_left = fopen(output_file_left, "wb");

  Rprintf("Saving the left  context word vectors \n");

  // Save the word vectors
  fprintf(fo_left, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo_left, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0l[a * layer1_size + b], sizeof(real), 1, fo_left);
    else for (b = 0; b < layer1_size; b++) fprintf(fo_left, "%lf ", syn0l[a * layer1_size + b]);
    fprintf(fo_left, "\n");
  }
  fclose(fo_left);

  Rprintf("Saving the right context word vectors \n");


  fo_right = fopen(output_file_right, "wb");
  // Save the word vectors
  fprintf(fo_right, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo_right, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0r[a * layer1_size + b], sizeof(real), 1, fo_right);
    else for (b = 0; b < layer1_size; b++) fprintf(fo_right, "%lf ", syn0r[a * layer1_size + b]);
    fprintf(fo_right, "\n");
  }
  fclose(fo_right);


  Rprintf("Saving the output word vectors \n");


  fo_out = fopen(output_file_out, "wb");
  // Save the word vectors
  fprintf(fo_out, "%lld %lld\n", vocab_size, layer1_size);

  if(negative == 0){
          for (a = 0; a < vocab_size; a++) {
          fprintf(fo_out, "%s ", vocab[a].word);
          if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo_out);
          else for (b = 0; b < layer1_size; b++) fprintf(fo_out, "%lf ", syn1[a * layer1_size + b]);
          fprintf(fo_out, "\n");
          }
          fclose(fo_out);
  }else{
          for (a = 0; a < vocab_size; a++) {
          fprintf(fo_out, "%s ", vocab[a].word);
          if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo_out);
          else for (b = 0; b < layer1_size; b++) fprintf(fo_out, "%lf ", syn1neg[a * layer1_size + b]);
          fprintf(fo_out, "\n");
          }
          fclose(fo_out);
  }

}



void train_word2vec2to1(char *train_file0, char *output_file0_left, char *output_file0_right, char *output_file0_out, 
					   char *binary0, char *dims0, char *threads,
                   	   char *window0, char *cbow0, char *min_count0, 
                   	   char *iter0, char *neg_samples0)
{
	int i;
    layer1_size = atoll(dims0);
    num_threads = atoi(threads);
    window=atoi(window0);
    binary = atoi(binary0);
	cbow = atoi(cbow0);
	min_count = atoi(min_count0);
	iter = atoll(iter0);
	negative = atoi(neg_samples0);
	strcpy(train_file, train_file0);
	strcpy(output_file_left, output_file0_left);
	strcpy(output_file_right, output_file0_right);
	strcpy(output_file_out, output_file0_out);


	alpha = 0.025;
	starting_alpha = alpha;
	word_count_actual = 0;

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
    	expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	TrainModel();
}
