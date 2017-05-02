#' @title Train a model by bi-directional word2vec (biword2vec).
#' @description Trains a bi-directional word2vec model on a corpus or training data, which is in general a .txt
#'              file. A bi-directional word2vec model uses separate vector representations for the left
#'              and right context of a word. This enables the user to determine enrichment of association
#'              of a word with another in terms of context of use.
##' @param train_file Path of a single .txt file for training. Tokens are split on spaces.
##' @param output_file_left Path of the output file for the left context words.
##' @param output_file_right Path of the output fle for the right context words.
##' @param vectors The number of vectors to output. Defaults to 100.
##' More vectors usually means more precision, but also more random error, higher memory usage, and slower operations.
##' Sensible choices are probably in the range 100-500.
##' @param threads Number of threads to run training process on.
##' Defaults to 1; up to the number of (virtual) cores on your machine may speed things up.
##' @param window The size of the window (in words) to use in training.
##' @param classes Number of classes for k-means clustering. Not documented/tested.
##' @param cbow If 1, use a continuous-bag-of-words model instead of skip-grams.
##' Defaults to false (recommended for newcomers).
##' @param min_count Minimum times a word must appear to be included in the samples.
##' High values help reduce model size.
##' @param iter Number of passes to make over the corpus in training.
##' @param force Whether to overwrite existing model files.
##' @param negative_samples Number of negative samples to take in skip-gram training. 0 means full sampling, while lower numbers
##' give faster training. For large corpora 2-5 may work; for smaller corpora, 5-15 is reasonable.
##' @return A VectorSpaceModel object.
##'
##' @references \url{https://code.google.com/p/word2vec/}
##' @export
##'
##' @import wordVectors
##' @useDynLib WEAVER
##'
##' @examples \dontrun{
##' model = train_biword2vec("nation.txt", output_file_left = "out_left.bin",
##'                          output_file_right = "out_right.bin", threads = 3,
##'                          woindow = 5)
##' }
train_biword2vec <- function(train_file, output_file_left = "vectors_left.bin", output_file_right = "vectors_right.bin",
                           vectors=100,threads=3,window=12,
                           classes=0,cbow=0,min_count=1,iter=5,force=F,
                           negative_samples=5)
{
  if (!file.exists(train_file)) stop("Can't find the training file!")
  if (file.exists(output_file_left) && !force) stop("The output file '",
                                     output_file_left ,
                                     "' already exists: give a new destination or run with 'force=TRUE'.")
  if (file.exists(output_file_right) && !force) stop("The output file '",
                                     output_file_right ,
                                     "' already exists: give a new destination or run with 'force=TRUE'.")

  train_dir <- dirname(train_file)

  # cat HDA15/data/Dickens/* | perl -pe 'print "1\t"' | egrep "[a-z]" | bookworm tokenize token_stream > ~/test.txt

  if(missing(output_file_left)) {
    output_file_left <- gsub(gsub("^.*\\.", "", basename(train_file)), "bin", basename(train_file))
    output_file_left <- file.path(train_dir, output_file_left)
  }

  if(missing(output_file_right)) {
    output_file_right <- gsub(gsub("^.*\\.", "", basename(train_file)), "bin", basename(train_file))
    output_file_right <- file.path(train_dir, output_file_right)
  }

  outfile_left_dir <- dirname(output_file_left)
  if (!file.exists(outfile_left_dir)) dir.create(outfile_left_dir, recursive = TRUE)

  outfile_right_dir <- dirname(output_file_right)
  if (!file.exists(outfile_right_dir)) dir.create(outfile_right_dir, recursive = TRUE)

  train_file <- normalizePath(train_file, winslash = "/", mustWork = FALSE)
  output_file_left <- normalizePath(output_file_left, winslash = "/", mustWork = FALSE)
  output_file_right <- normalizePath(output_file_right, winslash = "/", mustWork = FALSE)
  # Whether to output binary, default is 1 means binary.
  binary = 1

  OUT <- .C("biword2vec_wrap",
            train_file = as.character(train_file),
            output_file_left = as.character(output_file_left),
            output_file_right = as.character(output_file_right),
            binary = as.character(binary),
            dims=as.character(vectors),
            threads=as.character(threads),
            window=as.character(window),
            cbow=as.character(cbow),
            min_count=as.character(min_count),
            iter=as.character(iter),
            neg_samples=as.character(negative_samples)
  )

 ll <- list()
 ll[["left"]] <-  wordVectors::read.vectors(output_file_left)
 ll[["right"]] <- wordVectors::read.vectors(output_file_right)
 return(ll)
}
