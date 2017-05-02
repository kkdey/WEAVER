#' @title Word Set Enrichment Analysis (WSEA) across corpora
#'
#' @description Performs a word set enrichment across multiple corpora in relation to a
#'              topic of interest, specified by the \code{header_set}.
#'
#' @param model_list A list of VectorSpace models for a number of corpora. The length of the
#'                   list is equal to the number of corpora.
#' @param word_sets A list of vector of words (word sets) whose enrichment is to be studied.
#' @param header_set a word or a vector of words defining a topic of interest.
#' @param model_names The names of the corpora that are compared. Defaults to numerical groups.
#' @param add_groups A list of vectors indicating group labels that the user wants to compare.
#'                   Using \code{add_groups}, the user can compare a set of corpora with another
#'                   set of corpora beyod the 1v1 and 1vall comparisons the default version does.
#' @param numwords The number of neighboring words to each word in the word sets pooled into the
#'                 word set enrichment analysis.
#' @param fgsea.control The control parameters for the GSEA model fitting using the fgsea() package.
#'
#' @return Produces 1v1, 1vall WSEA output for all copora (each output similar to the fgsea output)
#'         together with group based WSEA output as specified by the \code{add_groups}.
#'
#' @importFrom wordVectors nearest_to
#' @importFrom wordVectors cosineSimilarity
#' @importFrom fgsea fgsea
#'
#' @export


wsea <-  function(model_list,
                  word_sets,
                  header_set,
                  model_names = NULL,
                  add_groups = NULL,
                  numwords = 1000,
                  fgsea.control = list()){

  fgsea.control.default <- list(minSize=5,
                                maxSize=500, nperm=10000,
                                nproc=0,
                                gseaParam=1,
                                BPPARAM=NULL)
  fgsea.control <- modifyList(fgsea.control.default, fgsea.control)


  if(is.null(model_names)){
    if(!is.null(names(model_list))){
      model_names <- names(model_list)
    }else{
      model_names <- 1:length(model_list)
    }
  }

  if(!is.null(add_groups)){
    if(!is.list(add_groups)){
      stop("add_groups must be a list if not NULL")
    }
  }

  if(is.null(word_sets)){
    stop("No word sets provided to perform WSEA")
  }

  if(is.null(header_set)){
    stop("header sets not provided: word group comparisons not possible")
  }

  if(!is.list(word_sets)){
    stop("word sets must be a list")
  }

  pooled_words <- unique(c(unlist(word_sets), header_set))

  listed_words <- as.character()

  for(k in 1:length(model_list)){
    for(m in 1:length(pooled_words)){
      listed_words <- c(listed_words, names(wordVectors::nearest_to(model_list[[k]],model_list[[k]][[paste0(pooled_words[m])]], numwords)))
    }
  }

  listed_words <- unique(listed_words)

  word_sets_extended <- vector(mode="list", length=length(word_sets))

  for(l in 1:length(word_sets)){
    for(h in 1:length(header_set)){
      word_sets_extended[[l]] <- c(word_sets_extended[[l]], paste0(word_sets[[l]], "-", header_set[h]))
    }
  }

  names <- c()
  for(h in 1:length(header_set)){
    names <- c(names, paste0(listed_words, "-", header_set[h]))
  }

  message("Building word2vec similarity matrix for related words")

  mat <- matrix(0, length(listed_words)*length(header_set), length(model_list))
  for(k in 1:length(model_list)){
    ## find cosine similarity of words from listed words occurring in model k
    temp1 <- as.data.frame(wordVectors::cosineSimilarity(model_list[[k]][[header_set, average = FALSE]], model_list[[k]][[listed_words, average = FALSE]]))
    ##  which words out of listed words occurred in the model k (kth newspaper)
    indices <- match(colnames(temp1), listed_words)
    ## flling the matrix with the cosine similarities
    for(i in 1:dim(temp1)[1]){
      slice <- mat[((i-1)*length(listed_words)+1): (i*length(listed_words)),k]
      slice[indices] <- as.numeric(unlist(temp1[i,]))
      mat[((i-1)*length(listed_words)+1): (i*length(listed_words)), k] <- slice
    }
  }

  num_zeros <- apply(mat, 1, function(x) return(length(which(x == 0))))
  bad_indices <- which(num_zeros >=2)
  rownames(mat) <- names

  ###  removing indices/words that do not occur in 2 or more newspapers  #####

  mat_filtered <- mat[-bad_indices, ]
  names_filtered <- names[-bad_indices]

  ### replacing NAs (at most 1 per row) by the row average

  for(k in 1:dim(mat_filtered)[1]){
    index <- which(mat_filtered[k, ] ==0)
    if(length(index)!=0){
      mat_filtered[k, index] <- mean(mat_filtered[k, -index])
    }
  }

  ###  performing quantile normalization

  message("Performing quantile normalization on the models")

  norm_mat_filtered <- matrix(0, dim(mat_filtered)[1], dim(mat_filtered)[2])
  norm_mat_filtered = preprocessCore::normalize.quantiles(as.matrix(mat_filtered))
  rownames(norm_mat_filtered) <- rownames(mat_filtered)


  #########  performing WSEA  ##########################

  message("Performing WSEA")

  norm_mat_names <- rownames(norm_mat_filtered)
  wsea <- list()

  message("Fitting the 1v1 WSEA")

  model_names <- c("The Nation", "Slate", "Root", "National Review")

  for(l in 2:dim(norm_mat_filtered)[2]){
    for(ll in 1:(l-1)){
      score <- as.vector(norm_mat_filtered[,l] - norm_mat_filtered[,ll])
      names(score) <- norm_mat_names
      wsea[[paste0("1v1: ", model_names[l], "-", model_names[ll])]] <- do.call(fgsea::fgsea, append(list(pathways = word_sets_extended,
                                                                                                         stats = score), fgsea.control))
      rm(score)
    }
  }

 message("Fitting the 1v-all WSEA")

  for(k in 1:dim(mat_filtered)[2]){
    temp_score_one_vs_all <- norm_mat_filtered[,k] - norm_mat_filtered[,-k]
    temp_score_min <- apply(temp_score_one_vs_all, 1, min)
    names(temp_score_min) <- norm_mat_names
    score <- (temp_score_min - mean(temp_score_min))/sd(temp_score_min)
    wsea[[paste0("1v-all: ", model_names[k])]] <- do.call(fgsea::fgsea, append(list(pathways = word_sets_extended,
                                                        stats = score), fgsea.control))
    rm(score)
  }

 if(!is.null(add_groups)){
   message("Fitting the added groups WSEA")
   for(m in 1:length(add_groups)){
     if(!is.vector(add_groups[[m]])){
       message("element of add_groups must be a vector")
     }else{
       temp_score_group <- norm_mat_filtered[,add_groups[[m]]] - norm_mat_filtered[,-add_groups[[m]]]
       temp_score_group_min <- apply(temp_score_group, 1, min)
       names(temp_score_group_min) <- norm_mat_names
       score <- (temp_score_group_min - mean(temp_score_group_min))/sd(temp_score_group_min)
       wsea[[paste0("group: ", m)]] <- do.call(fgsea::fgsea, append(list(pathways = word_sets_extended,
                                                                                                  stats = score), fgsea.control))
       rm(score)
     }
   }
 }

 message("Finished successfully !!")
 ww <- list("wsea" = wsea, "matrix" = norm_mat_filtered)
 return(ww)
}
