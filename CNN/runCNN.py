import sys,warnings
import numpy as np

from cnnModel import *
from loadWordVec import *

warnings.filterwarnings('ignore')
sys.setrecursionlimit(40000)

def parseSentence(text,wordIndex,maxLen):
	'''
	>>>convert sentence to a matrix

	>>>type text:string
	>>>para text:raw text
	>>>type wordIndex:dict
	>>>para wordIndex:map word to its entry
	>>>type maxLen:int
	>>>para maxLen:maximum length of sentences in the whole set
	'''
	length=len(text)
	padLeft=(maxLen-length+1)/2
	padRight=(maxLen-length)/2
	vec=[]

	for i in xrange(padLeft):
		vec.append(0)
	for word in text:
		vec.append(wordIndex[word])
	for i in xrange(padRight):
		vec.append(0)

	assert len(vec)==maxLen
	return vec

def loadDatas(dataFile,wordVecFile='',dimension=300,rand=False):
	'''
	>>>load training/validate/test data and wordVec info

	>>>type dataFile/wordVecFile: string
	>>>para dataFile/wordVecFile: data and wordVec file
	>>>type dimension: int
	>>>para dimension: the dimension of word embeddings
	>>>type static: bool
	>>>para static: static wordVec or not
	'''
	fopen=open(dataFile,'rb')
	data=cPickle.load(fopen)
	fopen.close()
	sentences,vocab,config=data
		
	wordVec=[]
	if rand==False:
		vectors,wordIndex=getWordVec(dataFile,wordVecFile)
	else:
		vectors,wordIndex=getRandWordVec(dataFile,dimension)

	return sentences,vocab,config,vectors,wordIndex

def parseConfig(sentences,vocab,config,vectors,wordIndex,static,name):
	'''
	>>>load configs to generate model and train/validate/test batches

	>>>sentences/vocab/config/vectors/wordIndex is the same in README.md file of each dataset
	>>>type static:bool
	>>>para static:whether or not to use static wordVec
	>>>type name:str
	>>>para name:model's name
	'''
	categories=config['classes']
	sets=config['all']
	train=config['train']
	validation=config['dev']
	test=config['test']
	cross=config['cross']
	dimension=len(vectors[0])
	batchSize=25

	maxLen=0
	for sentence in sentences:
		length=len(sentence['text'])
		if length>maxLen:
			maxLen=length

	setMatrix={}
	setClasses={}
	for subset in sets:
		setMatrix[subset]=[]
		setClasses[subset]=[]

	for sentence in sentences:
		vec=parseSentence(sentence['text'],wordIndex,maxLen)
		setLabel=sentence['setLabel']
		category=sentence['label']
		setMatrix[setLabel].append(vec)
		setClasses[setLabel].append(category)

	if cross==False:
		trainSet={};trainSetX=[];trainSetY=[]
		validateSet={};validationSetX=[];validationSetY=[]
		testSet={};testSetX=[];testSetY=[]
		for subset in train:
			trainSetX+=setMatrix[subset]
			trainSetY+=setClasses[subset]
		for subset in validation:
			validationSetX+=setMatrix[subset]
			validationSetY+=setClasses[subset]
		for subset in test:
			testSetX+=setMatrix[subset]
			testSetY+=setClasses[subset]

		if len(validation)==0:				#No ValidationSet
			newTrainSetX=[];newValidationSetX=[]
			newTrainSetY=[];newValidationSetY=[]
			index=0
			for i in np.random.permutation(range(len(trainSetX))):
				if index<len(trainSetX)*0.9:
					newTrainSetX.append(trainSetX[i])
					newTrainSetY.append(trainSetY[i])
				else:
					newValidationSetX.append(trainSetX[i])
					newValidationSetY.append(trainSetY[i])
				index+=1
			trainSetX=newTrainSetX;validationSetX=newValidationSetX
			trainSetY=newTrainSetY;validationSetY=newValidationSetY
				

		if len(trainSetX)%batchSize>0:
			extraNum=batchSize-len(trainSetX)%batchSize
			extraIndex=np.random.permutation(range(len(trainSetX)))
			for i in xrange(extraNum):
				trainSetX.append(trainSetX[extraIndex[i]])
				trainSetY.append(trainSetY[extraIndex[i]])

		if len(validationSetX)%batchSize>0:
			extraNum=batchSize-len(validationSetX)%batchSize
			extraIndex=np.random.permutation(range(len(validationSetX)))
			for i in xrange(extraNum):
				validationSetX.append(validationSetX[extraIndex[i]])
				validationSetY.append(validationSetY[extraIndex[i]])

		#trainSize=len(trainSetX)
		#validateSize=trainSize/5-(trainSize/5)%batchSize
		#validateIndex=np.random.permutation(range(trainSize))
		#for i in xrange(validateSize):
		#	validateSetX.append(trainSetX[validateIndex[i]])
		#	validateSetY.append(trainSetY[validateIndex[i]])

		trainSet['x']=np.array(trainSetX,dtype=theano.config.floatX)
		trainSet['y']=np.array(trainSetY,dtype=theano.config.floatX)
		validateSet['x']=np.array(validationSetX,dtype=theano.config.floatX)
		validateSet['y']=np.array(validationSetY,dtype=theano.config.floatX)
		testSet['x']=np.array(testSetX,dtype=theano.config.floatX)
		testSet['y']=np.array(testSetY,dtype=theano.config.floatX)

		network=CNNModel(
			wordMatrix=vectors,
			shape=(batchSize,1,maxLen,dimension),
			filters=(3,4,5),
			rfilter=(5,1),
			features=(100,),
			time=5,categories=categories,
			static=static,
			dropoutRate=(0.5,),
			learningRate=0.01,
			name=name
		)

		precision=network.train_validate_test(trainSet,validateSet,testSet,10)
		network.save()
		print 'Final Precision Rate %f%%'%(precision*100.)
	else:
		precisions=[]
		for item in sets:
			trainSet={};trainSetX=[];trainSetY=[]
			validateSet={};validationSetX=[];validationSetY=[]
			testSet={};testSetX=[];testSetY=[]
			for subset in sets:
				if item!=subset:
					trainSetX+=setMatrix[subset]
					trainSetY+=setClasses[subset]
				else:
					testSetX+=setMatrix[subset]
					testSetY+=setClasses[subset]

			#No ValidationSet
			newTrainSetX=[];newValidationSetX=[]
			newTrainSetY=[];newValidationSetY=[]
			index=0
			for i in np.random.permutation(range(len(trainSetX))):
				if index<len(trainSetX)*0.9:
					newTrainSetX.append(trainSetX[i])
					newTrainSetY.append(trainSetY[i])
				else:
					newValidationSetX.append(trainSetX[i])
					newValidationSetY.append(trainSetY[i])
				index+=1
			trainSetX=newTrainSetX;validationSetX=newValidationSetX
			trainSetY=newTrainSetY;validationSetY=newValidationSetY


			if len(trainSetX)%batchSize>0:
				extraNum=batchSize-len(trainSetX)%batchSize
				extraIndex=np.random.permutation(range(len(trainSetX)))
				for i in xrange(extraNum):
					trainSetX.append(trainSetX[extraIndex[i]])
					trainSetY.append(trainSetY[extraIndex[i]])

			if len(validationSetX)%batchSize>0:
				extraNum=batchSize-len(validationSetX)%batchSize
				extraIndex=np.random.permutation(range(len(validationSetX)))
				for i in xrange(extraNum):
					validationSetX.append(validationSetX[extraIndex[i]])
					validationSetY.append(validationSetY[extraIndex[i]])

			trainSet['x']=np.array(trainSetX,dtype=theano.config.floatX)
			trainSet['y']=np.array(trainSetY,dtype=theano.config.floatX)
			validateSet['x']=np.array(validationSetX,dtype=theano.config.floatX)
			validateSet['y']=np.array(validationSetY,dtype=theano.config.floatX)
			testSet['x']=np.array(testSetX,dtype=theano.config.floatX)
			testSet['y']=np.array(testSetY,dtype=theano.config.floatX)

			network=CNNModel(
				wordMatrix=vectors,
				shape=(batchSize,1,maxLen,dimension),
				filters=(3,4,5),
				rfilter=(5,1),
				features=(100,),
				time=1,categories=categories,
				static=static,
				dropoutRate=(0.5,),
				learningRate=0.01,
				name=name
			)

			precision=network.train_validate_test(trainSet,validateSet,testSet,10)
			network.save()
			precisions.append(precision)
		print 'Final Precision Rate %f%%'%(np.mean(precisions)*100.)

if __name__=='__main__':
	static=False
	rand=False
	mode=0
	dataFile=''
	vecFile=''
	name=''

	for i in xrange(len(sys.argv)):
		if i==0:
			continue
		if sys.argv[i]=='-d':
			mode=1
		elif sys.argv[i]=='-v':
			mode=2
		elif sys.argv[i]=='-n':
			mode=3
		else:
			if mode==1:
				dataFile=sys.argv[i]
				mode=0
			elif mode==2:
				vecFile=sys.argv[i]
				mode=0
			elif mode==3:
				name=sys.argv[i]
				mode=0
			else:
				if sys.argv[i]=='-nonstatic':
					static=False
				elif sys.argv[i]=='-static':
					static=True
				elif sys.argv[i]=='-rand':
					rand=True
				elif sys.argv[i]=='-word2vec':
					rand=False
				else:
					raise NotImplementedError('command line error')
	print 'config: dataFile:%s, vecFile:%s, static:%r, rand:%r'%(dataFile,vecFile,static,rand)

	sentences,vocab,config,vectors,wordIndex=loadDatas(dataFile=dataFile,wordVecFile=vecFile,dimension=300,rand=rand)
	parseConfig(sentences,vocab,config,vectors,wordIndex,static,name)
