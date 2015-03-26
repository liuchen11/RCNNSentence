import numpy as np

import theano
import theano.tensor as T

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,activation,dropout):
		'''
		>>>type rng: numpy.random.RandomState
		>>>para rng: initalize weight randomly

		>>>type input: theano.tensor.TensorType
		>>>para input: input data

		>>>type n_in: int
		>>>para n_in: the num of input neurons
		
		>>>type n_out: int
		>>>para n_out: the num of output neurons

		>>>type activation: func
		>>>para activation: the activate function

		>>>type dropout: boolean
		>>>para dropout: whether or not to use dropout
		'''
		self.input=input

		w_bound=np.sqrt(6.0/(n_in+n_out))

		w_value=np.asarray(
			rng.uniform(low=-w_bound,high=w_bound,size=(n_in,n_out)),
			dtype=theano.config.floatX
			)
		
		if activation==T.nnet.sigmoid:
			w_value*=4
		self.w=theano.shared(value=w_value,name='w',borrow=True)

		b_value=np.zeros((n_out),dtype=theano.config.floatX)
		self.b=theano.shared(value=b_value,name='b',borrow=True)

		raw_output=T.dot(input,self.w)+self.b

		self.output=(
			raw_output if activation is None
			else activation(raw_output)
			)

		# if dropout==True:
		# 	mask_vec=np.asarray(
		# 		rng.uniform(low=-10,high=10,size=(n_out)),
		# 		dtype=theano.config.floatX
		# 		)
		# 	for i in xrange(n_out):
		# 		if mask_vec[i]<0:
		# 			self.output[i]=0

		self.param=[self.w,self.b]
