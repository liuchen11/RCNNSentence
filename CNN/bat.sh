for i in 1 2 3 4 5
do
	THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python runCNN.py -n CNNMODEL-nonstatic -v ~/GoogleNews-vectors-negative300.bin -d ../SST1/data -nonstatic -word2vec
done
