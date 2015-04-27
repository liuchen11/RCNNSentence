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

def AdadeltaMomentumUpdate(params,cost,stepSize=1.0,momentum=0.9,rho=0.95,epsilon=1e-6,norm_lim=9):
    '''
    >>>adadelta-update with momentum
    >>>type params: tuple or list
    >>>para params: parameters
    >>>type cost: T.tensorType
    >>>para cost: goal to optimize
    >>>type stepSize: float
    >>>para stepSize: stepsize
    >>>type momentum: float
    >>>para momentum: momentum parameter
    >>>type rho: float
    >>>para rho: memory
    >>>type epsilon: float
    >>>para epsilon: avoid zero-division
    >>>type norm_lim: float
    >>>para norm_lim: normalization limit
    '''
    updates=OrderedDict({})
    grads=T.grad(cost,params)
    for param,grad in zip(params,grads):
        empty=np.zeros_like(param.get_value())
        oldGrad=theano.shared(value=as_floatX(empty),name='expGrad_%s'%param.name)
        oldUpdate=theano.shared(value=as_floatX(empty),name='expUpdate_%s'%param.name)
        oldDelta=theano.shared(value=as_floatX(empty),name='oldDelta_%s'%param.name)

        newGrad=rho*oldGrad+(1-rho)*T.sqr(grad)
        updates[oldGrad]=newGrad

        step=-(T.sqrt(oldUpdate+epsilon)/T.sqrt(newGrad+epsilon))*grad
        newDelta=step*stepSize
        updates[oldDelta]=newDelta
        newParam=param+newDelta+momentum*oldDelta

        newUpdate=rho*oldUpdate+(1-rho)*T.sqr(step)
        updates[oldUpdate]=newUpdate

        if param.get_value(borrow=True).ndim==2 and param.name!='wordVec':
            colNorm=T.sqrt(T.sum(T.sqr(newParam),axis=0))
            desiredNorm=T.clip(colNorm,0,T.sqrt(norm_lim))
            scale=desiredNorm/(1e-7+colNorm)
            updates[param]=newParam/scale
        else:
            updates[param]=newParam
    return updates

def AdadeltaUpdate(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
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

def sgdMomentum(params,cost,learningRate,momentum=0.9):
    '''
    >>>SGD optimizer with momentum
    >>>type params: tuple or list
    >>>para params: parameters of the model
    >>>type cost: T.tensorType
    >>>para cost: goal to be optimized
    >>>type learningRate: float
    >>>para learningRate: learning rate
    >>>type momentum: float
    >>>para momentum: momentum weight
    '''
    grads=T.grad(cost,params)
    updates=OrderedDict({})

    for param_i,grad_i in zip(params,grads):
        mparam_i=theano.shared(np.zeros(param_i.get_value().shape,dtype=theano.config.floatX),broadcastable=param_i.broadcastable)
        delta=momentum*mparam_i-learningRate*grad_i
        updates[mparam_i]=delta
        updates[param_i]=param_i+delta
    return updates

def sgd(params,cost,learningRate):
    '''
    >>>SGD Update
    >>>parameters are the same as above
    '''
    grads=T.grad(cost,params)
    updates=OrderedDict({})

    for param_i,grad_i in zip(params,grads):
        updates[param_i]=param_i-learningRate*grad_i
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
        self.lr=T.fscalar('lr')

        self.wordVec=theano.shared(wordMatrix,name='wordVec')
        input=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape(shape)

        self.deep=min(len(features),len(filters),len(poolSize))
        self.layers=[]
        print 'This is a network of %i layer(s)'%self.deep

        for i in xrange(self.deep):
            if i==0:
                layerSize=shape
                layerInput=input
                fmapIn=self.featureMaps
            else:
                layerSize=[self.batchSize,features[i-1],(self.layers[-1].shape[2]-filters[i-1][0]+1)/poolSize[i-1][0],(self.layers[-1].shape[3]-filters[i-1][1]+1)/poolSize[i-1][1]]
                layerInput=self.layers[-1].output
                fmapIn=features[i-1]
            newlayer=DropoutConvPool(
                    rng=rng,
                    input=layerInput,
                    shape=layerSize,
                    filters=[features[i],fmapIn,filters[i][0],filters[i][1]],
                    pool=poolSize[i],
                    dropout=dropoutRate[i]
                    )
            self.layers.append(newlayer)

        classifierInputShape=[self.batchSize,features[self.deep-1],(self.layers[-1].shape[2]-filters[self.deep-1][0]+1)/poolSize[self.deep-1][0],(self.layers[-1].shape[3]-filters[self.deep-1][1]+1)/poolSize[self.deep-1][1]]
        self.classifier=LogisticRegression(
                input=self.layers[-1].output.flatten(2),
                n_in=np.prod(classifierInputShape[1:]),
                n_out=categories
                )

        self.params=self.classifier.param
        for i in xrange(self.deep):
            self.params+=self.layers[i].param
        if static==False:
            self.params+=[self.wordVec]

        weights=0
        for item in self.classifier.param:
                weights+=T.sum(T.sqr(item))

        self.cost=self.classifier.negative_log_likelyhood(self.y)
        self.errors=self.classifier.errors(self.y)
        
        self.sgdUpdate=sgd(self.params,self.cost,self.lr)
        self.sgdMomentumUpdate=sgdMomentum(self.params,self.cost,self.lr)
        self.adadeltaUpdate=AdadeltaUpdate(self.params,self.cost)
        self.adadeltaMomentumUpdate=AdadeltaMomentumUpdate(params=self.params,cost=self.cost,stepSize=self.lr)

        self.sgdDelta=self.plotUpdate(self.sgdUpdate)
        self.sgdMomentumDelta=self.plotUpdate(self.sgdMomentumUpdate)
        self.adadeltaDelta=self.plotUpdate(self.adadeltaUpdate)
        self.adadeltaMomentumDelta=self.plotUpdate(self.adadeltaMomentumUpdate)

        print 'model %s constructed!'%name

    def plotUpdate(self,updates):
        '''
        >>>get update info of each layer
        >>>type updates: dict
        >>>para updates: update dictionary
        '''
        maxdict=T.zeros(shape=(self.deep*2+1,))
        mindict=T.zeros(shape=(self.deep*2+1,))
        meandict=T.zeros(shape=(self.deep*2+1,))
        
        for i in xrange(self.deep):
            updw=updates[self.layers[i].w]-self.layers[i].w
            maxdict=T.set_subtensor(maxdict[2*i],T.max(updw))
            mindict=T.set_subtensor(mindict[2*i],T.min(updw))
            meandict=T.set_subtensor(meandict[2*i],T.mean(updw))
            updb=updates[self.layers[i].b]-self.layers[i].b
            maxdict=T.set_subtensor(maxdict[2*i+1],T.max(updb))
            mindict=T.set_subtensor(mindict[2*i+1],T.min(updb))
            meandict=T.set_subtensor(meandict[2*i+1],T.mean(updb))

        updw=updates[self.classifier.w]-self.classifier.w
        maxdict=T.set_subtensor(maxdict[self.deep*2],T.max(updw))
        mindict=T.set_subtensor(mindict[self.deep*2],T.min(updw))
        meandict=T.set_subtensor(meandict[self.deep*2],T.mean(updw))
        return [maxdict,mindict,meandict]

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
        learnRate=T.fscalar('lr')
        stepSize=T.fscalar('lr')

        sgdTrainModel=theano.function(
                [index,learnRate],[self.cost,self.sgdDelta[0],self.sgdDelta[1],self.sgdDelta[2]],updates=self.sgdUpdate,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize],
                    self.lr:learnRate}
                )
        print 'SGD TrainModel Constructed!'

        sgdMomentumTrainModel=theano.function(
                [index,learnRate],[self.cost,self.sgdMomentumDelta[0],self.sgdMomentumDelta[1],self.sgdMomentumDelta[2]],updates=self.sgdMomentumUpdate,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize],
                    self.lr:learnRate}
                )
        print 'SGD-Momentum TrainModel Constructed!'

        adadeltaTrainModel=theano.function(
                [index],[self.cost,self.adadeltaDelta[0],self.adadeltaDelta[1],self.adadeltaDelta[2]],updates=self.adadeltaUpdate,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Adadelta TrainModel Constructed!'

        adadeltaMomentumTrainModel=theano.function(
                [index,stepSize],[self.cost,self.adadeltaMomentumDelta[0],self.adadeltaMomentumDelta[1],self.adadeltaMomentumDelta[2]],updates=self.adadeltaMomentumUpdate,
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize],
                    self.lr:stepSize}
                )
        print 'Adadelta(with momentum) TrainModel Constructed!'

        validateModel=theano.function(
                [index],self.errors,
                givens={
                    self.x:validateX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:validateY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Validation Model Constructed!'

        testTrain=theano.function(
                [index],[self.cost,self.errors],
                givens={
                    self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
                    self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]}
                )
        print 'Test Model on Training Set Constructed!'
       
        testInput=self.wordVec[T.cast(self.x.flatten(),dtype='int32')].reshape((testSize,self.featureMaps,self.sentenceLen,self.wdim))
        testOutput=0
        for i in xrange(self.deep):
            testOutput=self.layers[i].process(testInput,testSize)
            testInput=testOutput
        testClassifierInput=testInput.flatten(2)
        testPredict=self.classifier.predictInstance(testClassifierInput)
        testError=T.mean(T.neq(testPredict,self.y))
        testModel=theano.function([self.x,self.y],testError)
        print 'Testing Model Constructed!'

        epoch=0
        maxEpoch=5.0
        learningRate=self.learningRate
        steppingSize=1.0
        localOpt=0
        bestTestAcc=0.0
        bestValAcc=0.0
        finalAcc=0.0
        self.trainAccs=[]
        self.validateAccs=[]
        self.testAccs=[]
        self.costValues=[]
        self.result={}

        while epoch<nEpoch and epoch<maxEpoch:
            epoch+=1
            num=0

            for minBatch in np.random.permutation(range(trainBatches)):
                cost,dmax,dmin,dmean=adadeltaTrainModel(minBatch)
                #cost=adadeltaMomentumTrainModel(minBatch,steppingSize)
                x=float(epoch)+float(num+1)/float(trainBatches)-1
                #self.costValues.append({'x':x,'value':cost})
                #if num%10==0:
                #    print 'max:',dmax
                #    print 'min:',dmin
                #    print 'mean:',dmean
                if num%50==0:
                    trainResult=[testTrain(i) for i in xrange(trainBatches)]
                    trainCost,trainError=np.mean(trainResult,axis=0)
                    trainAcc=1-trainError
                    self.costValues.append({'x':x,'value':trainCost})
                    validateError=[validateModel(i) for i in xrange(validateBatches)]
                    validateAcc=1-np.mean(validateError)
                    self.trainAccs.append({'x':x,'acc':trainAcc})
                    self.validateAccs.append({'x':x,'acc':validateAcc})
                    print'Epoch=%i,Num=%i,TrainAcc=%f%%,ValidateAcc=%f%%'%(epoch,num,trainAcc*100.,validateAcc*100.)

                    if validateAcc>bestValAcc:
                        testError=testModel(testX,testY)
                        testAcc=1-testError
                        bestValAcc=validateAcc
                        bestTestAcc=max(bestTestAcc,testAcc)
                        finalAcc=testAcc
                        self.testAccs.append({'x':x,'acc':testAcc})
                        print 'TestAcc=%f%%'%(testAcc*100.)
                        localOpt=0
                        maxEpoch=max(maxEpoch,epoch*1.5)
                    else:
                        localOpt+=1
                        if localOpt>=5:
                            learningRate/=10
                            steppingSize/=2
                            localOpt=0
                            print 'Learning Rate %f->%f'%(learningRate*10.,learningRate)
                            print 'stepping Size %f->%f'%(steppingSize*2.,steppingSize)
                    print 'BestValAcc=%f%%,BestTestAcc=%f%%,FinalAcc=%f%%'%(bestValAcc*100.,bestTestAcc*100.,finalAcc*100.)
                num+=1

            x=float(epoch)
            trainResult=[testTrain(i) for i in xrange(trainBatches)]
            trainCost,trainError=np.mean(trainResult,axis=0)
            trainAcc=1-trainError
            self.costValues.append({'x':x,'value':trainCost})
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
                localOpt=0
                maxEpoch=max(maxEpoch,epoch*1.5)
            else:
                localOpt+=1
                if localOpt>=5:
                    learningRate/=10
                    steppingSize/=2
                    localOpt=0
                    print 'Learning Rate %f->%f'%(learningRate*10.,learningRate)
                    print 'Stepping Size %f->%f'%(steppingSize*2.,steppingSize)
            print 'BestValAcc=%f%%,BestTestAcc=%f%%,FinalAcc=%f%%'%(bestValAcc*100.,bestTestAcc*100.,finalAcc*100.)

        self.result={'minError':1-bestTestAcc,'finalAcc':finalAcc,'bestValAcc':bestValAcc}
        return finalAcc

    def save(self):
        savePath='../Results/'
        timeStruct=time.localtime(time.time())
        fileName=str(timeStruct.tm_mon)+'_'+str(timeStruct.tm_mday)+'_'+str(timeStruct.tm_hour)+'_'+str(timeStruct.tm_min)+'__'+str(self.result['finalAcc'])+'_'+self.name
        cPickle.dump([self.result,self.trainAccs,self.validateAccs,self.testAccs,self.costValues],open(savePath+fileName,'wb'))
