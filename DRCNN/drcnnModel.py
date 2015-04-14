import sys,cPickle
import time
import numpy as np
import theano
import theano.tensor as T

from collections import defaultdict,OrderedDict
from loadWordVec import *
from hiddenLayer import *
from logisticRegression import *
from normLayer import *
from recurrentConvLayer import *
from convLayer import *

sys.setrecursionlimit(40000)

def ReLU(x):
	return T.switch(x>0,x,0)

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
    >>>type norm_lim:int
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

class DRCNNModel(object):

    def __init__(self,wordMatrix,shape,filters,rfilter,features,poolSize,time,categories,static,dropoutRate,learningRate,name):
        '''
        >>>initialize the model
        
        >>>type wordMatrix: matrix
        >>>para wordMatrix: input tensor
        >>>type shape: tuple or list of length 4
        >>>para shape: (batchSize,feature maps,sentenceLen,dimension)
        >>>type filters: tuple or list of 2-len tuple or list
        >>>para filters: the size of filters in each layer
        >>>type rfilter: tuple or list of 2-len tuple or list
        >>>para rfilter: the size of recurrent connection in each layer
        >>>type features: tuple or list of int
        >>>para features: num of feature maps in each layer
        >>>type poolSize: tuple or list of 2-len tuple or list
        >>>para poolSize: pooling size of each layer
        >>>type time: int
        >>>para time: the iteration times of recurrent connection
        >>>type categories: int
        >>>para categories: target categories
        >>>type static: boolean
        >>>para static: static wordVec or not
        >>>type dropoutRate: tuple or list of float
        >>>para dropoutRate: dropout rate of each layer
        >>>type learningRate: float
        >>>para learningRate: learning rate
        >>>type name: str
        >>>para name: the name of the model
        '''
        self.learningRate=learningRate
        self.static=static
        self.name=name
        self.batchSize,self.featureMaps,self.sentenceLen,self.wdim=shape

        rng=np.random.RandomState(2011010539)
        
        self.x=T.matrix('x')
        self.y=T.ivector('y')
        self.lr=T.dscalar('lr')

        self.wordVec=theano.shared(wordMatrix,name='wordVec')
        input=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape(shape)

        layer0InputShape=shape
        self.layer0=DropoutConvPool(
                rng=rng,
                input=input,
                shape=layer0InputShape,
                filters=[features[0],self.featureMaps,filters[0][0],filters[0][1]],
                pool=poolSize[0],
                dropout=dropoutRate[0]
                )

        layer1InputShape=[self.batchSize,features[0],(layer0InputShape[2]-filters[0][0]+1)/poolSize[0][0],(layer0InputShape[3]-filters[0][1]+1)/poolSize[0][1]]
        self.layer1=DropoutConvPool(
                rng=rng,
                input=self.layer0.output,
                shape=layer1InputShape,
                filters=[features[1],features[0],filters[1][0],filters[1][1]],
                pool=poolSize[1],
                dropout=dropoutRate[1]
                )

        layer2InputShape=[self.batchSize,features[1],(layer1InputShape[2]-filters[1][0]+1)/poolSize[1][0],(layer1InputShape[3]-filters[1][1]+1)/poolSize[1][1]]
        self.layer2=DropoutConvPool(
                rng=rng,
                input=self.layer1.output,
                shape=layer2InputShape,
                filters=[features[2],features[1],filters[2][0],filters[2][1]],
                pool=poolSize[2],
                dropout=dropoutRate[2]
                )

        classifierInputShape=[self.batchSize,features[2],(layer2InputShape[2]-filters[2][0]+1)/poolSize[2][0],(layer2InputShape[3]-filters[2][1]+1)/poolSize[2][1]]
        self.classifier=LogisticRegression(
                input=self.layer2.output.flatten(2),
                n_in=np.prod(classifierInputShape[1:]),
                n_out=categories
                )

        self.params=self.layer0.param+self.layer1.param+self.layer2.param+self.classifier.param
        if static==False:
            self.params+=[self.wordVec]

        weights=0
        for item in self.classifier.param:
                weights+=T.sum(T.sqr(item))

        self.cost=self.classifier.negative_log_likelyhood(self.y)+1e-4*weights
        self.errors=self.classifier.errors(self.y)
        grads=T.grad(self.cost,self.params)
        self.update=[
                (param_i,param_i-self.lr*grad_i)
                for (param_i,grad_i) in zip(self.params,grads)
                ]
        
        self.adadeltaUpdate=AdadeltaUpdate(self.params,self.cost)

        print 'model %s constructed!'%name

    def train_validate_test(self,trainSet,validateSet,testSet,nEpoch):
        '''
        >>>train and test the model

        >>>type trainSet/validateSet/testSet: dict
        >>>para trainSet/validateSet/testSet: train/validate/test set
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
        learnRate=T.dscalar('lr')

        sgdTrainModel=theano.function(
                [index,learnRate],self.cost,updates=self.update,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize],
                    self.lr:learnRate}
                )
        print 'SGD TrainModel Constructed!'
        adadeltaTrainModel=theano.function(
                [index],self.cost,updates=self.adadeltaUpdate,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Adadelta TrainModel Constructed!'

        validateModel=theano.function(
                [index],self.errors,
                givens={
                    self.x:validateX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:validateY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Validation Model Constructed!'

        testTrain=theano.function(
                [index],self.errors,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Test Model on Training Set Constructed!'
       
        testLayer0Input=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape((testSize,self.featureMaps,self.sentenceLen,self.wdim))
        testLayer1Input=self.layer0.process(testLayer0Input,testSize)
        testLayer2Input=self.layer1.process(testLayer1Input,testSize)
        testClassifierInput=self.layer2.process(testLayer2Input,testSize).flatten(2)
        testPredict=self.classifier.predictInstance(testClassifierInput)
        testError=T.mean(T.neq(testPredict,self.y))
        testModel=theano.function([self.x,self.y],testError)
        print 'Testing Model Constructed!'

        epoch=0
        learningRate=self.learningRate
        bestTestAcc=0.0
        bestValAcc=0.0
        finalAcc=0.0
        self.trainAccs=[]
        self.validateAccs=[]
        self.testAccs=[]
        self.costValues=[]
        self.result={}

        while epoch<nEpoch:
            epoch+=1
            num=0

            for minBatch in np.random.permutation(range(trainBatches)):
                cost=adadeltaTrainModel(minBatch)
                x=float(epoch)+float(num+1)/float(trainBatches)-1
                self.costValues.append({'x':x,'value':cost})
                if num%50==0:
                    trainError=[testTrain(i) for i in xrange(trainBatches)]
                    trainAcc=1-np.mean(trainError)
                    validateError=[validateModel(i) for i in xrange(validateBatches)]
                    validateAcc=1-np.mean(validateError)
                    self.trainAccs.append({'x':x,'acc':trainAcc})
                    self.validateAccs.append({'x':x,'acc':validateAcc})
                    print'Epoch=%i,TrainAcc=%f%%,ValidateAcc=%f%%'%(epoch,trainAcc*100.,validateAcc*100.)

                    if validateAcc>bestValAcc:
                        testError=testModel(testX,testY)
                        testAcc=1-testError
                        bestValAcc=validateAcc
                        bestTestAcc=max(bestTestAcc,testAcc)
                        finalAcc=testAcc
                        self.testAccs.append({'x':x,'acc':testAcc})
                        print 'TestAcc=%f%%'%(testAcc*100.)
                    print 'BestValAcc=%f%%,BestTestAcc=%f%%,FinalAcc=%f%%'%(bestValAcc,bestTestAcc,finalAcc)
                num+=1

            x=float(epoch)
            trainError=[testTrain(i) for i in xrange(trainBatches)]
            trainAcc=1-np.mean(trainError)
            validateError=[validateModel(i) for i in xrange(validateBatches)]
            validateAcc=1-np.mean(validateError)
            self.trainAccs.append({'x':x,'acc':trainAcc})
            self.validateAccs.append({'x':x,'acc':validateAcc})
            print 'Epoch=%i,TrainAcc=%f%%,ValidateAcc=%f%%'%(epoch,trainAcc*100.,validateAcc*100.)

            if validateAcc>bestValAcc:
                testError=testModel(testX,testY)
                testAcc=1-testError
                bestValAcc=validateAcc
                bestTestAcc=max(bestTestAcc,testAcc)
                finalAcc=testAcc
                self.testAccs.append({'x':x,'acc':testAcc})
                print 'TestAcc=%f%%'%(testAcc*100.)
            print 'BestValAcc=%f%%,BestTestAcc=%f%%,FinalAcc=%f%%'%(bestValAcc,bestTestAcc,finalAcc)

        self.result={'minError':1-bestTestAcc,'finalAcc':finalAcc,'bestValAcc':bestValAcc}
        return finalAcc

    def save(self):
        savePath='../Results/'
        timeStruct=time.localtime(time.time())
        fileName=str(timeStruct.tm_mon)+'_'+str(timeStruct.tm_mday)+'_'+str(timeStruct.tm_hour)+'_'+str(timeStruct.tm_min)+'__'+str(self.result['finalAcc'])+'_'+self.name
        cPickle.dump([self.result,self.trainAccs,self.validateAccs,self.testAccs,self.costValues],open(savePath+fileName,'wb'))
