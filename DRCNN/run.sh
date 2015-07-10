THEANO_FLAGS=optimizer_excluding=fusion:inplace,profile=True python runDRCNN.py -n 3L-RCNNPOOL -v ~/GoogleNews-vectors-negative300.bin -d ../MR/data -static -word2vec
