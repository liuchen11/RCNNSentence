import cPickle
import numpy as np

def loadBinVec(fileName,vocab)
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
				ch=fread(1)
				if ch==' ':
					word=''.join(word)
					break
				else:
					word.append(ch)
			if word in vocab:
				wordVec[word]=np.fromstring(f.read(binaryLen),dtype='float32')
			else
				f.read(binaryLen)
	return layerSize,wordVec

def getWordVec(configFileName,vecFileName)
	'''
	>>>load wordvecs and initialize unknow word's vector randomly
	'''
	data=cPickle.load(open(configFileName,'rb'))
	dimension,wordVec=loadBinVec(vecFileName,data[1])
	for word in data[1]:
		if word not in wordVec:
			wordVec[word]=np.random.uniform(-0.25,0.25,dimension)
	return dimension,wordVec

def getRandWordVec(configFileName,dimension)
	'''
	>>>initialize the word vectors randomly given a dimension value
	'''
	data=cPickle.load(open(configFileName,'rb'))
	wordVec={}
	for word in data[1]:
		wordVec[word]=np.random.uniform(-0.25,0.25,dimension)
	return dimension,wordVec
