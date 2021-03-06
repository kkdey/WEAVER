% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/w2vec_net.R
\name{w2vec_net}
\alias{w2vec_net}
\title{Network Visualization of the context of a word in a corpus.}
\usage{
w2vec_net(model, header, numwords_header = 20, numwords_sec = 5,
  value_scale = 1000, node_color = "red", node_shape = "dot",
  node_shadow = TRUE)
}
\arguments{
\item{model}{A word2vec VectorSpace model ouput.}

\item{header}{A word whose context the user wants to plot in the network.}

\item{numwords_header}{The number of words primarily linked (nearest) to the header word
(with the highest cosine similairities) taken in the network.
Defaults to 20.}

\item{value_scale}{The scale for the size of the nodes in the network, with the size of the nodes
being proportional to how closely they are associated (in terms of
cosine similairty) with the header. Defaults to 1000.}

\item{node_color}{The color of the nodes in the network. Defaults to red color.}

\item{node_shadow}{Whether shadowin is used for the nodes. Defaults to TRUE.}

\item{numwors_sec}{The number of secondary words connected to each of the primary link words
(highest cosine similarities with the primary link words). Defaults to 5.}

\item{node_shaoe}{The shape of the nodes in the networl. Defaults to "dot".}
}
\value{
Returns an interactive network visulaization of the \code{header} word along with
        its primary and secondary linked words.
}
\description{
displays an interactive network visualization of the context of a word in a corpus.
             The network visualization uses the \code{visNetwork()} package.
}
