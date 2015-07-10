#!/usr/bin/python2.7
# -*- coding: utf-8 -*- 
import cPickle
import re

from collections import defaultdict

def cleanStr(string):
	string=re.sub(r'^A-Za-z0-9(),!?\'\`',' ',string)
	string=re.sub(r'\s{2,}',' ',string)
	string=string.replace('Ã¡','á').replace('Ã©','é').replace('Ã±','ñ').replace('Â','').replace('Ã¯','ï')
	string=string.replace('Ã¼','ü').replace('Ã¢','â').replace('Ã¨','è').replace('Ã¶','ö').replace('Ã¦','æ')
	string=string.replace('Ã³','ó').replace('Ã»','û').replace('Ã´','ô').replace('Ã£','ã').replace('Ã§','ç')
	string=string.replace('Ã  ','à ').replace('Ã','í').replace('í­','í')
	return string

def loadSentences(fileName):
	Index2Sentence={}
	Sentence2Index={}
	with open(fileName,'r') as fopen:
		i=0
		for line in fopen:
			if i>0:
				parts=line.split('\t')
				index=int(parts[0])
				sentence=parts[1].replace('-LRB-','(').replace('-RRB-',')').replace('\n','')
				sentence=cleanStr(sentence)				
				Index2Sentence[index]=sentence
				Sentence2Index[sentence]=index
			i+=1
	return Index2Sentence, Sentence2Index

def lookupDict(dictFileName,Sentence2Index):
	Sentence2SentimentIndex={}
	with open(dictFileName,'r') as fopen:
		for line in fopen:
			parts=line.split('|')
			sentiment=parts[0].replace('-LRB-','(').replace('-RRB-',')').replace('\n','')
			sentiment=cleanStr(sentiment)
			index=int(parts[1])
			if sentiment in Sentence2Index:
				Sentence2SentimentIndex[sentiment]=index
	assert len(Sentence2SentimentIndex)==len(Sentence2Index)
	for sentence in Sentence2Index:
		if not sentence in Sentence2SentimentIndex:
			print sentence
	return Sentence2SentimentIndex

def loadLabels(sentimentLabelFile):
	SentimentIndex2Label={}
	with open(sentimentLabelFile,'r') as fopen:
		i=0
		for line in fopen:
			if i>0:
				parts=line.split('|')
				index=int(parts[0])
				value=min(int(float(parts[1])//0.2),4)
				SentimentIndex2Label[index]=value
			i+=1
	return SentimentIndex2Label

def loadSetLabel(setLabelFile):
	'''
	>>>SetLabel: 1-train 2-test 3-dev
	'''
	Index2SetLabel={}
	with open(setLabelFile,'r') as fopen:
		i=0
		for line in fopen:
			if i>0:
				parts=line.split(',')
				index=int(parts[0])
				setLabel=int(parts[1])
				Index2SetLabel[index]=setLabel
			i+=1
	return Index2SetLabel

def loadData(Sentence2Index,Index2Sentence,Sentence2SentimentIndex,SentimentIndex2Label,Index2SetLabel):
	vocab=defaultdict(float)
	sentences=[]
	for sentence in Sentence2Index:
		index=Sentence2Index[sentence]
		sentimentIndex=Sentence2SentimentIndex[sentence]
		label=SentimentIndex2Label[sentimentIndex]
		setLabel=Index2SetLabel[index]
		clean=cleanStr(sentence)
                clean=clean.lower()
		words=set(clean.split())
		for word in words:
			vocab[word]+=1
		sentences.append({'label':label,'text':clean.split(),'setLabel':setLabel,'len':len(clean.split())})
	return sentences,vocab

path='./'

if __name__=='__main__':
	fileName=path+'datasetSentences.txt'
	dictFileName=path+'dictionary.txt'
	sentimentLabelFile=path+'sentiment_labels.txt'
	setLabelFile=path+'datasetSplit.txt'
	Index2Sentence,Sentence2Index=loadSentences(fileName)
	Sentence2SentimentIndex=lookupDict(dictFileName,Sentence2Index)
	SentimentIndex2Label=loadLabels(sentimentLabelFile)
	Index2SetLabel=loadSetLabel(setLabelFile)
	sentences,vocab=loadData(Sentence2Index,Index2Sentence,Sentence2SentimentIndex,SentimentIndex2Label,Index2SetLabel)
	cPickle.dump([sentences,vocab,{'classes':5,'all':[1,2,3],'train':[1],'test':[2],'dev':[3],'cross':False}],open('data','wb'))
	print 'data processed'
