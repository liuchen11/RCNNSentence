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
	with open(positiveFile,'r') as fopen:
		for line in fopen:
			clean=cleanStr(line)
			words=set(clean.split())
			for word in words:
				vocab[word]+=1
			sentences.append({'label':0,'text':clean,'setLabel':np.random.randint(0,10),'len':len(clean)})
	with open(negativeFile,'r') as fopen:
		for line in fopen:
			clean=cleanStr(line)
			words=set(clean.split())
			for word in words:
				vocab[word]+=1
			sentences.append({'label':1,'text':clean,'setLabel':np.random.randint(0,10),'len':len(clean)})
	return sentences,vocab

path='./'

if __name__=='__main__':
	positiveFile=path+'rt-polarity.pos'
	negativeFile=path+'rt-polarity.neg'
	sentences,vocab=loadSentences(positiveFile,negativeFile)
	cPickle.dump([sentences,vocab],open('data','wb'))
