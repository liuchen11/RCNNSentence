import cPickle
import numpy as np
import sys

if __name__=='__main__':
    if len(sys.argv)<2:
        print 'Usage: python showErrors.py <errorFile>'
        exit(0)

    errorInfo=cPickle.load(open(sys.argv[1],'rb'))
    
    indexWord=errorInfo['indexWord']
    testSet=errorInfo['testSet']
    predictMatrix=errorInfo['predictMatrix']
    testPredict=errorInfo['testPredict']
    categories=len(predictMatrix)
    names=[]
    if categories==5:
        names=['negative','slightly negative','neutral','slightly positive','positive']
    elif categories==2:
        names=['negative','positive']
    else:
        for i in xrange(categories):
            names.append(str(i))
    testX=np.asarray(testSet['x'],dtype='int32')
    testY=np.asarray(testSet['y'],dtype='int32')
    testPred=np.asarray(testPredict,dtype='int32')

    while True:
        print 'type in \'print\' to overview or \'real -> predict\' to make a query or \'exit\' to exit'
        query=raw_input('>>>')

        if query=='exit':
            break
        if query=='print':
            print 'R/P',
            for i in xrange(categories):
                print '\t',i,
            print '\n',
            for i in xrange(categories):
                print i,
                for j in xrange(categories):
                    print '\t',predictMatrix[i,j],
                print '\n',
        else:
            try:
                parts=query.split()
                assert parts[1]=='->'
                real=int(parts[0])
                predict=int(parts[2])
                print 'real:%s -> predict:%s, %i instances'%(names[real],names[predict],predictMatrix[real,predict])
                index=1
                for i in xrange(len(testY)):
                    if testY[i]==real and testPred[i]==predict:
                        print index,'\t',
                        for j in xrange(len(testX[i])):
                            if testX[i][j]!=0:
                                print indexWord[testX[i][j]],
                        print '\n',
                        index+=1
            except:
                print 'unknown command'
