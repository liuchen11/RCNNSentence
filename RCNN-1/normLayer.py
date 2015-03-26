import numpy as np

import theano
import theano.tensor as T

class NormLayer(object):
	
	def __init__(self,input,shape,alpha,beta,N):
		'''
		>>>type input: T.tensor4
		>>>para input: input tensor

		>>>type shape: tuple or list of length 4
		>>>para shape: (batch_size,features,height,width)

		>>>type alpha/beta/N: float/float/int
		>>>para alpha/beta/N: normalization factors
		'''
		
		tmp=input.dimshuffle(1,0,2,3)
		self.output=T.zeros_like(tmp)
		for k in xrange(shape[1]):
			result=T.sum(T.square(tmp[k-N/2:k+N/2]),axis=0)
			norm=(1.0+alpha/N*result)**beta
			self.output=T.set_subtensor(self.output[k],tmp[k]/norm)
		self.output=self.output.dimshuffle(1,0,2,3)
		
