

#################  testing the biword2vec model  #############################

library(wordVectors)
out <- WEAVER::train_biword2vec("../../BLM_project/the_nation_traindata.txt",
                                    output_file_left = "out_left.bin",
                                    output_file_right = "out_right.bin",
                                    output_file_out = "out_out.bin",
                                    cbow = 0, threads = 10,
                                    force = TRUE, min_count = 5,
                                    window = 10)

nearest_to(out$in_left, out$in_left[["movement"]], 30)
nearest_to(out$in_right, out$in_right[["movement"]], 30)
nearest_to(out$out, out$out[["movement"]], 30)

nearest_to(out$in_left, out$in_left[["lives"]], 10)
nearest_to(out$in_right, out$in_right[["lives"]], 10)
nearest_to(out$out, out$out[["lives"]], 10)

nearest_to(out$in_left, out$in_left[["police"]], 30)
nearest_to(out$in_right, out$in_right[["police"]], 30)
nearest_to(out$out, out$out[["police"]], 30)


out1 <- WEAVER::train_word2vec("../../BLM_project/the_nation_traindata.txt",
                               output_file_in = "out1_in.bin",
                               output_file_out = "out1_out.bin",
                               cbow = 0, force = TRUE, threads=10,
                               min_count = 5,
                               window = 10)

wordVectors::nearest_to(out1$`in`, out1$`in`[["lives"]], 20)
wordVectors::nearest_to(out1$out, out1$out[["lives"]], 20)

out2 <- wordVectors::train_word2vec("../../BLM_project/the_nation_traindata.txt",
                               output_file = "out_w2.bin",
                               cbow = 0, force = TRUE, threads=10,
                               min_count = 5,
                               window = 10)

wordVectors::nearest_to(out2, out2[["lives"]], 20)
wordVectors::nearest_to(out2, out2[["matter"]], 20)



which(names(wordVectors::nearest_to(out1, out1[["movement"]], 1000)) == "matter")
cosineSimilarity(out1[["matter"]], out1[["movement"]])

wordVectors::nearest_to(out1, out1[["lives"]], 30)

###########  Reading the word2vec models for left and right context vectors ######

out_left <- wordVectors::read.vectors("out_left.bin")
out_right <- wordVectors::read.vectors("out_right.bin")
