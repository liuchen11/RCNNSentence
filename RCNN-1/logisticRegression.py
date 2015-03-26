import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import softmax

class LogisticRegression(object):

	def __init__(self,input,n_in,n_out):
		'''
		>>>type input: T.TensorType
		>>>para input: input data

		>>>type n_in: int
		>>>para n_in: num of input neurons

		>>>type n_out: int
		>>>para n_out: num of output neurons
		'''
		self.w=theano.shared(
			value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
			name='w',
			borrow=True
			)
		self.b=theano.shared(
			value=np.zeros((n_out,),dtype=theano.config.floatX),
			name='b',
			borrow=True
			)
		self.param=[self.w,self.b]

		self.output=softmax(T.dot(input,self.w)+self.b)
		self.predict=T.argmax(self.output,axis=1)

	def negative_log_likelyhood(self,y):
		'''
		>>>calculate the negative log_likelyhood given labels of instances

		>>>type y: T.ivector
		>>>para y: right labels of instances
		'''
		return -T.mean(T.log(self.output)[T.arange(y.shape[0]),y])

	def errors(self,y):
		'''
		>>>calculate the error rate of test instances

		>>>type y: T.ivector
		>>>para y: right labels of instances
		'''
		return T.mean(T.neq(self.predict,y))

