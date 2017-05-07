#include "R.h"
#include "Rmath.h"
#include "word2vec.h"

void train_word2vec(char *train_file0, char *output_file0_in, char *output_file0_out,
                   char *binary20, char *dims0, char *threads2,
                   char *window20, char *cbow20,
                   char *min_count20, char *iter20, char *neg_samples20)
{
	  int i;
    layer1_size2 = atoll(dims0);
    num_threads2 = atoi(threads2);
    window2=atoi(window20);
	  binary2 = atoi(binary20);
	  cbow2 = atoi(cbow20);
	  min_count2 = atoi(min_count20);
	  iter2 = atoll(iter20);
	  negative2 = atoi(neg_samples20);
	  strcpy(train_file, train_file0);
	  strcpy(output_file_in, output_file0_in);
	  strcpy(output_file_out, output_file0_out);


	alpha2 = 0.025;
	starting_alpha2 = alpha2;
	word_count_actual2 = 0;

	vocab2 = (struct vocab_word2 *)calloc(vocab_max_size2, sizeof(struct vocab_word2));
	vocab_hash2 = (int *)calloc(vocab_hash_size2, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
    	expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	TrainModel2();
}


void word2vec_wrap(char **train_file, char **output_file_in, char **output_file_out,
                   char **binary2, char **dims, char **threads2,
                   char **window2, char **cbow2, char **min_count2, char **iter2,
                   char **neg_samples2)
{
    train_word2vec(*train_file, *output_file_in, *output_file_out,  *binary2, *dims,
                  *threads2, *window2, *cbow2,*min_count2,*iter2, *neg_samples2);
}

