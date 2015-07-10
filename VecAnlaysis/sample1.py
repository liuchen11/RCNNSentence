import numpy as np
import cPickle
import sys
import random

from loadWordVec import *

if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Usage: python sample.py <dataFile> <iterations>'
    
    vecFile=sys.argv[1]
    iterations=int(sys.argv[2])

    vectors,wordIndex=getWordVec('../SST1/data',vecFile)
    vocabSize=len(vectors)

    print 'loaded! %i words'%vocabSize
    
    sim=[]
    for i in xrange(iterations):
        w1=random.randint(0,vocabSize-1)
        w2=random.randint(0,vocabSize-1)
        vec1=vectors[w1]
        vec2=vectors[w2]
        cosine=np.dot(vec1,vec2)
        sim.append(abs(cosine))
        if (i+1)%100==0:
            avrg=np.mean(sim)
            print '%i iterations:%f'%(i+1,avrg)

    avrg=np.mean(sim)
    print 'final: iterations:%f'%avrg

