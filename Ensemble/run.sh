THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python runRCNN.py -n Ensemble -v ~/GoogleNews-vectors-negative300.bin -d ../SST1/data -nonstatic -word2vec
