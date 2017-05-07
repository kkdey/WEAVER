# WEAVER

Word Enrichment Analysis using VEctor Representations

**Author**:  [Kushal K Dey](http://kkdey.github.io/), [Matthew Stephens](http://stephenslab.uchicago.edu/)

## Summary

The main contents of this package are functions to do the following. 

- *Robust word2vec*  - adaptive shrinkage analysis of word2vec cosine similarities 
                       to provide robust estimates 

- *Network visualization* - network visualization of the context of a word in a corpus 
                            based on word2vec ouput.
                            
- *WSEA* - Word set enrichment analysis across multiple corpora for a topic of interest.

- *WSSA* - Word set significance analysis for a topic of interest for a given corpus.

- *Directional word2vec* - A direction based word2vec model implementation that separates out 
                           the vector representations of the left and right contexts of a word.                               Highlights enrichment of words in either sides of the context.
             
## Licenses

The WEAVER package is distributed under [GPL - General Public License (>= 2)]

## Contact

For any questions or comments, please contact [kkdey@uchicago.edu](kkdey@uchicago.edu)

## Acknowledgements

I would like to thank Benjamin Schmidt for the incredible **wordVectors** package which
acted as a principal guide in my venture into NLP. This package and many of its functions draw from his package.

I would also like to acknowledge Richard J. So, Lei Sun, Matthew Stephens, Michael Dawson and Allison Garner for helpful discussions and many of the functions in this package developed from the questions and comments they posed in front of me.

