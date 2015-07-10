import numpy as np
import cPickle
import sys
import random

def loadBinVec(fileName):
    wordVec={}
    wordIndex=[]
    with open(fileName,'rb') as fopen:
        header=fopen.readline()
        vocabSize,layerSize=map(int,header.split())
        binaryLen=np.dtype('float32').itemsize*layerSize
        for line in xrange(vocabSize):
            if line%10000==0:
                print '%i/%i'%(line,vocabSize)
            word=[]
            while True:
                ch=fopen.read(1)
                if ch==' ':
                    word=''.join(word)
                    break
                else:
                    word.append(ch)
            wordVec[word]=np.fromstring(fopen.read(binaryLen),dtype='float32')
            wordVec[word]/=np.linalg.norm(wordVec[word])
            wordIndex.append(word)
    return wordVec,wordIndex

if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Usage: python sample.py <dataFile> <iterations>'
    
    vecFile=sys.argv[1]
    iterations=int(sys.argv[2])

    wordVec,wordIndex=loadBinVec(vecFile)
    vocabSize=len(wordVec)

    print 'loaded! %i words'%vocabSize
    
    sim=[]
    for i in xrange(iterations):
        w1=random.randint(0,vocabSize-1)
        w2=random.randint(0,vocabSize-1)
        vec1=wordVec[wordIndex[w1]]
        vec2=wordVec[wordIndex[w2]]
        cosine=np.dot(vec1,vec2)
        sim.append((cosine+1)/2.0)
        if (i+1)%100==0:
            avrg=np.mean(sim)
            print '%i iterations:%f'%(i+1,avrg)

    avrg=np.mean(sim)
    print 'final: iterations:%f'%avrg

