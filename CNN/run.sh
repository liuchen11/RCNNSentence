THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python runCNN.py -n BaselineCNN -v ~/GoogleNews-vectors-negative300.bin -d ../SST1/data -nonstatic -word2vec
