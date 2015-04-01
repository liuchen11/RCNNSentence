import cPickle
import numpy as np
import theano
import theano.tensor as T
from collections import defaultdict, OrderedDict

from loadWordVec import *
from hiddenLayer import *
from convLayer import *
from logisticRegression import *
from normLayer import *
from recurrentConvLayer import *

def ReLU(x):
	return T.switch(x<0,0,x)

def as_floatX(variable):
	if isinstance(variable,float) or isinstance(variable,np.ndarray):
		return np.cast[theano.config.floatX](variable)
	return T.cast(variable,theano.config.floatX)

def AdadeltaUpdate(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
	'''
	>>>

	>>>type params: tuple or list
	>>>para params: parameters
	>>>type cost:
	>>>para cost:
	>>>type rho: float
	>>>para rho:
	>>>type epsilon: float
	>>>para epsilon:
	>>>type norm_lim: int
	>>>para norm_lim:
	'''
	updates=OrderedDict({})
	exp_sqr_grads=OrderedDict({})
	exp_sqr_update=OrderedDict({})
	g_params=[]
	for param in params:
		empty=np.zeros_like(param.get_value())
		exp_sqr_grads[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
		exp_sqr_update[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
		gp=T.grad(cost,param)
		g_params.append(gp)
	for param,gp in zip(params,g_params):
		exp_sg=exp_sqr_grads[param]
		exp_su=exp_sqr_update[param]
		update_exp_sg=rho*exp_sg+(1-rho)*T.sqr(gp)#????
		updates[exp_sg]=update_exp_sg
		
		step=-(T.sqrt(exp_su+epsilon)/T.sqrt(update_exp_sg+epsilon))*gp		
		stepped_param=param+step

		update_exp_su=rho*exp_su+(1-rho)*T.sqr(step)
		updates[exp_su]=update_exp_su

		if param.get_value(borrow=True).ndim==2 and param.name!='wordVec':
			col_norms=T.sqrt(T.sum(T.sqr(stepped_param),axis=0))
			desired_norms=T.clip(col_norms,0,T.sqrt(norm_lim))#???
			scale=desired_norms/(1e-7+col_norms)
			updates[param]=stepped_param*scale
		else:
			updates[param]=stepped_param
	return updates

class model(object):
	
	def __init__(self,wordMatrix,shape,filters,rfilter,features,time,
			categories,static,dropoutRate,learningRate):
		'''
		>>>initalize the model

		>>>type wordMatrix: matrix
		>>>para wordMatrix: input tensor
		>>>type shape: tuple or list of length 4
		>>>para shape: [batchSize,featureMaps,sentenceLen,dimension]
		>>>type filters: tuple or list of int
		>>>para filters: different sizes of filters
		>>>type rfilter: tuple of list of length 2
		>>>para rfilter: the filter size of recurrent connection
		>>>type features: tuple or list of int
		>>>para features: num of feature maps in each layer
		>>>type time: int
		>>>para time: the iteration times of recurrent connection
		>>>type categories: int
		>>>para categories: target categories
		>>>type static: boolean
		>>>para static: static wordVec or not
		>>>type dropoutRate: tuple of list of float
		>>>para dropoutRate: dropout rate of each layer
		>>>type learningRate: float
		>>>para learningRate: learning rate
		'''
		self.learningRate=learningRate
		self.static=static

		rng=np.random.RandomState(2011010539)
		self.batchSize,featureMaps,self.sentenceLen,self.dimension=shape

		filterSizes=[]
		poolSizes=[]
		for filter in filters:
			filterSizes.append([features[0],featureMaps,filter,self.dimension])
			poolSizes.append([self.sentenceLen-filter+1,1])

		#build up the model
		self.x=T.matrix('x')		#batch sentences
		self.y=T.ivector('y')		#output labels
		self.lr=T.dscalar('lr')

		self.wordVec=theano.shared(wordMatrix,name='wordVec')

		input=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape(shape)

		self.layers0=[]
		layer1Inputs=[]
		for i in xrange(len(filters)):
			filterSize=filterSizes[i]
			poolSize=poolSizes[i]
			RConvLayer=RecurrentConvLayer(
				rng=rng,
				input=input,
				shape=shape,
				filters=filterSize,
				rfilter=[features[0],features[0],rfilter[0],rfilter[1]],
				alpha=0.001, beta=0.75,
				N=int(features[0]/8+1),
				time=time,
				pool=poolSize
			)
			self.layers0.append(RConvLayer)
			layer1Inputs.append(RConvLayer.output.flatten(2))

		#self.layer1=DropoutHiddenLayer(
		#	rng,
		#	input=T.concatenate(layer1Inputs,1),
		#	n_in=len(filters)*features[0],
		#	n_out=categories,
		#	activation=ReLU,
		#	dropoutRate=dropoutRate[0]
		#)

		self.layer1=LogisticRegression(
			input=T.concatenate(layer1Inputs,1),
			n_in=len(filters)*features[0],
			n_out=categories,
		)

		self.cost=self.layer1.negative_log_likelyhood(self.y)
		self.errors=self.layer1.errors(self.y)

		self.params=self.layer1.param
		for layer in self.layers0:
			self.params+=layer.param
		if static==False:
			self.params+=[self.wordVec]

		grads=T.grad(self.cost,self.params)
		self.update=[
			(paramI,paramI-gradI*self.lr)
			for (paramI,gradI) in zip(self.params,grads)
		]
		self.adadeltaUpdate=AdadeltaUpdate(self.params,self.cost)

		print 'the model constructed!'

	def train_validate_test(self,trainSet,validateSet,testSet,nEpoch):
		'''
		>>>train and test the model

		>>>type trainSet/validateSet/testSet: matrix
		>>>para trainSet/validateSet/testSet: different subset

		>>>type nEpoch: int
		>>>para nEpoch: maximum iteration epoches
		'''
		trainSize=trainSet['x'].shape[0]
		validateSize=validateSet['x'].shape[0]
		testSize=testSet['x'].shape[0]
		trainX=theano.shared(trainSet['x'],borrow=True)
		trainY=theano.shared(trainSet['y'],borrow=True)
		trainY=T.cast(trainY,'int32')
		validateX=theano.shared(validateSet['x'],borrow=True)
		validateY=theano.shared(validateSet['y'],borrow=True)
		validateY=T.cast(validateY,'int32')
		testX=testSet['x']
		testY=np.asarray(testSet['y'],'int32')
		trainBatches=trainSize/self.batchSize
		validateBatches=validateSize/self.batchSize

		index=T.iscalar('index')
		testMatrix=T.matrix('WordMatrix')
		testLabel=T.iscalar('TestLabel')
		learnRate=T.scalar('lr')

		trainModel=theano.function(
		[index],self.cost,updates=self.adadeltaUpdate,
		givens={
		self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
		self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'training model constructed!'

		validateModel=theano.function(
		[index],self.errors,
		givens={
		self.x:validateX[index*self.batchSize:(index+1)*self.batchSize],
		self.y:validateY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'validation model constructed!'

		testLayer0Output=[]
		testLayer0Input=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape((testSize,1,self.sentenceLen,self.dimension))
		for layer in self.layers0:
			output=layer.process(testLayer0Input,testSize)
			testLayer0Output.append(output.flatten(2))
		testLayer1Input=T.concatenate(testLayer0Output,1)
		testPredict=self.layer1.predictInstance(testLayer1Input)
		testError=T.mean(T.neq(testPredict,self.y))
		testModel=theano.function([self.x,self.y],testError)
		print 'testing model constructed!'

		epoch=0
		iteration=0
		maxIteration=10000
		minError=1.0
		rate=0.01

		while epoch<nEpoch and iteration<maxIteration:
			epoch+=1
			if minError>0.25:
				rate=0.01
			elif minError>0.2:
				rate=0.005
			elif minError>0.25:
				rate=0.002
			else:
				rate=0.001

			for minBatch in np.random.permutation(range(trainBatches)):
				print 'training %i/%i'%(minBatch,trainBatches)
				cost=trainModel(minBatch)				#set zero func

			validateError=[
				validateModel(i)
				for i in xrange(validateBatches)
			]
			print 'testing...'
			validatePrecision=1-np.mean(validateError)
			testError=testModel(testX,testY)
			testPrecision=1-testError
			minError=min(minError,testError)

			print 'epoch=%i, validate precision %f%%, test precision %f%%'%(epoch,validatePrecision,testPrecision)
			print 'minError=%f%%'%(minError)

		return minError

