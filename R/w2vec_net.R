#' @title Network Visualization of the context of a word in a corpus.
#'
#' @description displays an interactive network visualization of the context of a word in a corpus.
#'              The network visualization uses the \code{visNetwork()} package.
#'
#' @param model A word2vec VectorSpace model ouput.
#' @param header A word whose context the user wants to plot in the network.
#' @param numwords_header The number of words primarily linked (nearest) to the header word
#'                        (with the highest cosine similairities) taken in the network.
#'                        Defaults to 20.
#' @param numwors_sec The number of secondary words connected to each of the primary link words
#'                    (highest cosine similarities with the primary link words). Defaults to 5.
#' @param value_scale The scale for the size of the nodes in the network, with the size of the nodes
#'                    being proportional to how closely they are associated (in terms of
#'                    cosine similairty) with the header. Defaults to 1000.
#' @param node_color The color of the nodes in the network. Defaults to red color.
#' @param node_shaoe The shape of the nodes in the networl. Defaults to "dot".
#' @param node_shadow Whether shadowin is used for the nodes. Defaults to TRUE.
#'
#' @return Returns an interactive network visulaization of the \code{header} word along with
#'         its primary and secondary linked words.
#'
#' @import visNetwork
#' @importFrom wordVectors nearest_to
#' @importFrom wordVectors cosineSimilarity
#'
#' @export

w2vec_net <- function(model,
                      header,
                      numwords_header = 20,
                      numwords_sec = 5,
                      value_scale = 1000,
                      node_color = "red",
                      node_shape = "dot",
                      node_shadow = TRUE){
  word_nbrs <- names(wordVectors::nearest_to(model,model[[paste0(header)]], numwords_header))
  secondary_words <- c()
  for(l in 1:length(word_nbrs)){
    secondary_words <- c(secondary_words, names(wordVectors::nearest_to(model,model[[paste0(word_nbrs[l])]], numwords_sec)))
  }


  node_words <- unique(c(word_nbrs, secondary_words))
  temp1 <- as.data.frame(wordVectors::cosineSimilarity(model[[node_words, average = FALSE]], model[[node_words, average = FALSE]]))
  diag(temp1) <- 0
  tt <- apply(temp1, 2, function(x) {
    index1 <- order(x, decreasing = TRUE)[1:5]
    y <- array(0, length(x))
    y[index1] <- x[index1]
    y[-index1] <- 0
    return(y)})

  for(i in 2:dim(tt)[1]){
    for(j in 1:(i-1)){
      if(tt[i,j] > 0){
        tt[j,i] = tt[i,j]
      }else if (tt[j,i] > 0){
        tt[i,j] = tt[j,i]
      }
    }
  }

  weights <- abs(tt)
  rownames(weights) <- colnames(weights)

  gg <- reshape2::melt(as.matrix(weights))
  dim(gg)
  gg <- gg[which(gg[,3] != 0), ]

  x1 <- factor(gg[,1], levels = node_words)
  x2 <- factor(gg[,2], levels = node_words)

  edges <- data.frame(from = as.numeric(x1),
                      to = as.numeric(x2))

  temp2 <- as.data.frame(cosineSimilarity(model[[node_words, average = FALSE]], model[[paste0(header), average = FALSE]]))
  indices <- match(node_words, rownames(temp2))
  names_old <- rownames(temp2)
  temp2_new <- temp2[indices, ]
  names(temp2_new) <- names_old[indices]

  nodes <- data.frame(id = 1:length(node_words), label = paste0(node_words),
                      value = value_scale*(as.numeric(temp2_new)),
                      color = node_color,
                      shape = node_shape,
                      shadow = node_shadow)
  visNetwork::visNetwork(nodes, edges, height = "800px", width = "100%") %>%
    visOptions(highlightNearest = TRUE,
               nodesIdSelection = TRUE) %>%
    visEdges(smooth = FALSE)
}
