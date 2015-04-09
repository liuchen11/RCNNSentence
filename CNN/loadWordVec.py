import cPickle
import theano
import numpy as np

def loadBinVec(fileName,vocab):
	'''
	>>>load *.bin file and return word embeddings of the vocabulary, and its dimension
	'''
	wordVec={}
	with open(fileName,'rb') as fopen:
		header=fopen.readline()
		vocabSize,layerSize=map(int,header.split())
		binaryLen=np.dtype('float32').itemsize*layerSize
		for line in xrange(vocabSize):
			word=[]
			while True:
				ch=fopen.read(1)
				if ch==' ':
					word=''.join(word)
					break
				else:
					word.append(ch)
			if word in vocab:
				wordVec[word]=np.fromstring(fopen.read(binaryLen),dtype='float32')
			else:
				fopen.read(binaryLen)
	return layerSize,wordVec

def getWordVec(configFileName,vecFileName):
	'''
	>>>load wordvecs and initialize unknow word's vector randomly
	'''
	num=0
	data=cPickle.load(open(configFileName,'rb'))
	dimension,wordVec=loadBinVec(vecFileName,data[1])
	for word in data[1]:
		if word not in wordVec:
			num+=1
			wordVec[word]=np.random.uniform(-0.25,0.25,dimension)

	vocabSize=len(data[1])
	vectors=np.zeros(shape=(vocabSize+1,dimension))
	wordIndex={}
	vectors[0]=np.zeros(dimension)
	index=1
	for word in wordVec:
		vectors[index]=wordVec[word]
		wordIndex[word]=index
		index+=1
	print 'word not found: ',num
	return vectors,wordIndex

def getRandWordVec(configFileName,dimension):
	'''
	>>>initialize the word vectors randomly given a dimension value
	'''
	data=cPickle.load(open(configFileName,'rb'))
	wordVec={}
	for word in data[1]:
		wordVec[word]=np.random.uniform(-0.25,0.25,dimension)

	vocabSize=len(data[1])
	vectors=np.zeros(shape=(vocabSize+1,dimension))
	wordIndex={}
	vectors[0]=np.zeros(dimension)
	index=1
	for word in wordVec:
		vectors[index]=wordVec[word]
		wordIndex[word]=index
		index+=1
	return vectors,wordIndex
