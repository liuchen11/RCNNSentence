THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python runDCNN.py -n 3LDropout -v ~/GoogleNews-vectors-negative300.bin -d ../SST1/data -nonstatic -word2vec
