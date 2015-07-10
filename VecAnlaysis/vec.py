import cPickle
import sys,warnings
import scipy
import scipy.misc
import numpy as np

from loadWordVec import *

warnings.filterwarnings('ignore')

def parseSentence(text,wordIndex,vectors,maxLen):
    length=len(text)
    padLeft=(maxLen-length+1)/2
    padRight=(maxLen-length)/2
    dimension=len(vectors[0])
    vec=[]

    for i in xrange(padLeft):
        vec.append(np.zeros(shape=(dimension,)))
    for word in text:
        index=wordIndex[word]
        norm=np.linalg.norm(vectors[index])
        vec.append(vectors[index]/norm)
    for i in xrange(padRight):
        vec.append(np.zeros(shape=(dimension,)))

    assert len(vec)==maxLen
    return np.asarray(vec)

def loadData(dataFile,wordVecFile,outputFile):
    fopen=open(dataFile,'rb')
    data=cPickle.load(fopen)
    fopen.close()
    sentences,vocab,config=data

    vectors,wordIndex=getWordVec(dataFile,wordVecFile)

    maxLen=0
    for sentence in sentences:
        length=len(sentence['text'])
        if length>maxLen:
            maxLen=length
    dimension=len(vectors[0])

    stat=np.zeros(shape=(maxLen,11))
    avrg=[]
    for i in xrange(maxLen):
        avrg.append([])
    pic_matrix=np.zeros(shape=(10,maxLen))
    for i in xrange(len(sentences)):
        print '%i/%i'%(i,len(sentences))
        sentence=sentences[i]
        vecs=parseSentence(sentence['text'],wordIndex,vectors,maxLen)
        co_matrix=np.zeros(shape=(maxLen,maxLen))
        low=(maxLen+1-len(sentence['text']))/2
        high=len(sentence['text'])+low
        for j in xrange(maxLen):
            for k in xrange(maxLen):
                if k>=low and k<high and j>=low and j<high:
                    #co_matrix[j,k]=(np.dot(vecs[j],vecs[k])+1)/2.0
                    co_matrix[j,k]=np.abs(np.dot(vecs[j],vecs[k]))
                    if j>=k:
                        level=int(co_matrix[j,k]//0.1)
                        stat[j-k,level]+=1
                        avrg[j-k].append(co_matrix[j,k])
                else:
                    co_matrix[j,k]=0.0
        pic_matrix=np.concatenate((pic_matrix,co_matrix,np.zeros(shape=(10,maxLen))),axis=0)  
        if i%200==0:
            scipy.misc.imsave(str(i/200)+outputFile,pic_matrix)
            pic_matrix=np.zeros(shape=(10,maxLen))
            print '%i-%i printed'%(i-200,i)

    for i in xrange(maxLen):
        if len(avrg[i])!=0:
            avrg[i]=np.mean(avrg[i])
        else:
            avrg[i]=0
    
    return avrg,stat

if __name__=='__main__':
    avrg,stat=loadData('../SST1/data','/home/liuchen/GoogleNews-vectors-negative300.bin','abs_show.jpg')
    cPickle.dump([avrg,stat],open('stat','wb'))
    print stat
    print avrg
