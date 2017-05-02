#include "R.h"
#include "Rmath.h"
#include "bi-word2vec.h"

void train_biword2vec(char *train_file0, char *output_file0_left, char *output_file0_right, char *binary0, char *dims0, char *threads,
                   char *window0, char *cbow0, char *min_count0, char *iter0, char *neg_samples0)
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


void biword2vec_wrap(char **train_file, char **output_file_left, char **output_file_right, char **binary, char **dims, char **threads,
                       char **window, char **cbow, char **min_count, char **iter, char **neg_samples)
{
    train_biword2vec(*train_file, *output_file_left, *output_file_right,  *binary, *dims, *threads,
    				*window,*cbow,*min_count,*iter, *neg_samples);
}

