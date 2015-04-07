import cPickle
import re
import numpy as np

from collections import defaultdict

def cleanStr(string):
	string=re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ",string)
	string=re.sub(r"\'s"," \'s",string)
	string=re.sub(r"\'ve"," \'ve",string)
	string=re.sub(r"n\'t"," n\'t",string)
	string=re.sub(r"\'re"," \'re",string)
	string=re.sub(r"\'d"," \'d",string)
	string=re.sub(r"\'ll"," \'ll",string)
	string=re.sub(r","," , ",string)
	string=re.sub(r"!"," ! ",string)
	string=re.sub(r"\("," \( ",string)
	string=re.sub(r"\)"," \) ",string)
	string=re.sub(r"\?"," \? ",string)
	string=re.sub(r"\s{2,}"," ",string)
	return string.strip().lower()

def loadSentences(positiveFile,negativeFile):
	sentences=[]
	vocab=defaultdict(float)
	poslines=[];neglines=[]
	with open(positiveFile,'r') as fopen:
		for line in fopen:
			poslines.append(line)
	posNum=len(poslines)
	rand=np.random.permutation(range(posNum))
	for i in xrange(posNum):
		clean=cleanStr(poslines[rand[i]])
		words=set(clean.split())
		for word in words:
			vocab[word]+=1
		sentences.append({'label':0,'text':clean.split(),'setLabel':i%10,'len':len(clean.split())})

	with open(negativeFile,'r') as fopen:
		for line in fopen:
			neglines.append(line)
	negNum=len(neglines)
	rand=np.random.permutation(range(negNum))
	for i in xrange(negNum):
		clean=cleanStr(neglines[rand[i]])
		words=set(clean.split())
		for word in words:
			vocab[word]+=1
		sentences.append({'label':1,'text':clean.split(),'setLabel':i%10,'len':len(clean.split())})
	return sentences,vocab

path='./'

if __name__=='__main__':
	positiveFile=path+'rt-polarity.pos'
	negativeFile=path+'rt-polarity.neg'
	sentences,vocab=loadSentences(positiveFile,negativeFile)
	cPickle.dump([sentences,vocab,{'classes':2,'all':range(10),'train':[],'test':[],'dev':[],'cross':True}],open('data','wb'))
	print 'data processed'
