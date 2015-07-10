THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python runRCNN.py -n RCNN3 -v ~/GoogleNews-vectors-negative300.bin -d ../MR/data -static -word2vec
