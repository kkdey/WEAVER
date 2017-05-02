#' @title Adaptive Shrinkage of cosine similarities in word2vec output.
#'
#' @description Performs an adaptive shrinkage (ash) of the cosine similarities in word2vec output.
#'              The ash framework has been proposed by Matthew Stephens (2016).
#'
#'
#' @param model_true A word2vec VectorSpace model output obtained by fitting the model on corpus.
#' @param model_boot_list A list of VectorSpace models obtained by fitting the word2vec on resampled
#'                        corpus data.
#' @param word_vec A word set or a vector of words of interest. Shrinkage of cosine similarities
#'                 for the primary linked words to each word in this word set.
#' @param num_related_words The number of primary linked words to each word taken into the model.
#'
#' @return Returns a list with each element corresponding to ash results for each word
#'        in \code{word_vec}. An element of the list is anopther list containing the following
#'        features.
#'             \code{similar_words} : all the words taken into the shrinkage model for
#'                                    each word in \code{word_vec}.
#'              \code{cosine_est}: cosine similarities from the original word2vec model
#'              \code{ sd_cosine_transform_est}: standard error of Fisher z-scores obtained from
#'                                       cosine similarities of primary links with each word
#'                                       in \code{word_vec} from resampled corpus data.
#'              \code{ash_out}: The ash result for each word in \code{word_vec}
#'               \code{ash_cosine_est}: ash shrunk cosine similarities for each word in
#'                                      \code{word_vec}.
#'
#' @importFrom wordVectors cosineSimilarity
#' @importFrom wordVectors nearest_to
#' @import ashr
#'
#' @export


corshrink_word2vec <- function(model_true, model_boot_list,
                               word_vec, num_related_words = 500){

  ############  We initialize the list which we will return at end containing the results
  ##########  from the analysis

   return_list <- vector(mode = "list", length(word_vec))

  ###########  the number of bootstrap samples : equal to length of model_boot_list  #####


    numboot <- length(model_boot_list)

  ############  for each word, carry out Corshrink over related words  ##################

    for(numvec in 1:length(word_vec)){
      ############  taking a word from the word vector passed
         word <- word_vec[numvec]

         ############  finding the neighbors or related words to that word  #############


         word_nbrs <- as.character()
         for(ll in 1:numboot){
           word_nbrs <- c(word_nbrs, names(wordVectors::nearest_to(model_boot_list[[ll]],
                                                      model_boot_list[[ll]][[paste0(word)]], num_related_words)))
         }

         listed_words <- unique(word_nbrs)
         listed_words <- listed_words[-1]

         ############ sd of cosine-similarity of related words to the given word
         ############  from bootstrap samples   #################

         cosine_transform_sd_vec <- array(0, length(listed_words))
         cosine_boot_extended <- array(0, length(listed_words))

         mat_temp <- matrix(0, numboot, length(listed_words))

         for(i in 1:numboot){
           cosine_boot <- unlist(as.data.frame(wordVectors::cosineSimilarity(model_boot_list[[i]][[word, average = FALSE]],model_boot_list[[i]][[listed_words, average = FALSE]])))
           match_indices <- match(names(cosine_boot), listed_words)
           cosine_boot_extended[match_indices] <- cosine_boot
           cosine_transform_boot_extended <- 0.5*log((1+cosine_boot_extended)/(1-cosine_boot_extended))
           mat_temp[i,] <- cosine_transform_boot_extended
        }

         cosine_transform_sd_vec <- apply(mat_temp, 2, function(x) return(sd(x)))

         cosine_est_extended <- array(0, length(listed_words))

         cosine_est <- unlist(as.data.frame(wordVectors::cosineSimilarity(model_true[[word, average = FALSE]], model_true[[listed_words, average = FALSE]])))
         match_indices <- match(names(cosine_est), listed_words)
         cosine_est_extended[match_indices] <- cosine_est
         cosine_transform_est_extended <- 0.5*log((1+cosine_est_extended)/(1-cosine_est_extended))
         names(cosine_est_extended) <- listed_words

         ash_out <- ashr::ash(as.vector(cosine_transform_est_extended), as.vector(cosine_transform_sd_vec))
         fitted_cos <- (exp(2*ash_out$result$PosteriorMean)-1)/(exp(2*ash_out$result$PosteriorMean)+1)
         names(fitted_cos) <- listed_words
         return_list[[numvec]] = list(similar_words = listed_words,
                                      cosine_est = cosine_est_extended,
                                      sd_cosine_transform_est = cosine_transform_sd_vec,
                                      ash_result = ash_out,
                                      ash_cosine_est = fitted_cos)
         cat("Processed the ash shrinkage of word2vec for word: ", paste0(word), "\n")
    }
    return(return_list)
}
