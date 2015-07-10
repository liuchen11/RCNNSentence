import cPickle
import numpy as np

data=cPickle.load(open('data','rb'))

sentences=data[0]
vocab=data[1]
config=data[2]
trainSentences=[]
categories=config['classes']

if config['cross']==False:
    trainLabel=config['train']
    for sentence in sentences:
        if sentence['setLabel'] in trainLabel:
            trainSentences.append(sentence)
else:
    trainSentences=sentences

print 'total sentences: %i'%(len(trainSentences))

while True:
    def findW(query):
        times=[0]*categories
        docs=[0]*categories

        for sentence in trainSentences:
            appear=sentence['text'].count(query)
            if appear>0:
                label=sentence['label']
                docs[label]+=1
                times[label]+=appear
        
        return docs,times

    print 'type in a word or nothing to exit'
    query=raw_input('>>>')
    
    if query=='' or query==None:
        break
    if query=='*':
        l=[]
        precessed=0
        for word in vocab:
            precessed+=1
            if precessed%100==0:
                print '%i/%i'%(precessed,len(vocab))
            docs,times=findW(word)
            if np.sum(times)<=10:
                continue
            l.append((word,times))
        l.sort(lambda x,y:cmp(float(x[1][3]+x[1][4]+1)/float(x[1][0]+x[1][1]+1),float(y[1][3]+y[1][4]+1)/float(y[1][0]+y[1][1]+1)))
        print '-------postive--------'
        for i in xrange(10):
            query=l[i][0]
            times=l[i][1]
            print 'word \'%s\' appears %i times in total'%(query,np.sum(times))
            print 'times-classes distribution',[times[0]+times[1],times[2],times[3]+times[4]]
        print '-------negative-------'
        for i in xrange(10):
            query=l[-i-1][0]
            times=l[-i-1][1]
            print 'word \'%s\' appears %i times in total'%(query,np.sum(times))
            print 'times-classes distribution',[times[0]+times[1],times[2],times[3]+times[4]]
        continue
    else:
        docs,times=findW(query)
        print 'word \'%s\' appears in %i sentences, %i times in total'%(query,np.sum(docs),np.sum(times))
        print 'sentences-classes distribution',[docs[0]+docs[1],docs[2],docs[3]+docs[4]]
        print 'times-classes distribution',[times[0]+times[1],times[2],times[3]+times[4]]
