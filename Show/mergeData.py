import sys
import cPickle

if __name__=='__main__':
    if len(sys.argv)<3:
        print 'Usage: python mergeData.py <files to merge>... <outputfile>'
        exit()
    
    files2merge=sys.argv[1:-1]
    outputfile=sys.argv[-1]

    epoch=0
    result={'finalAcc':0.0,'bestValAcc':0.0,'minError':1.0};costValue=[]
    trainAcc=[];validateAcc=[];testAcc=[]
    for f in files2merge:
        print 'loading file: '+f
        data=cPickle.load(open(f,'rb'))
        fresult=data[0];fcostValue=data[4]
        ftrainAcc=data[1];fvalidateAcc=data[2];ftestAcc=data[3]
        maxEpoch=epoch

        if fresult['bestValAcc']>=result['bestValAcc']:
            result['bestValAcc']=fresult['bestValAcc']
            result['finalAcc']=fresult['finalAcc']
            result['minError']=min(result['minError'],fresult['minError'])
        print 'result loaded!'

        for item in fcostValue:
            costValue.append({'x':item['x']+epoch,'value':item['value']})
            if item['x']+epoch>maxEpoch:
                maxEpoch=item['x']+epoch
        print 'costValue loaded!'

        for item in ftrainAcc:
            trainAcc.append({'x':item['x']+epoch,'acc':item['acc']})
            if item['x']+epoch>maxEpoch:
                maxEpoch=item['x']+epoch
        print 'trainAcc loaded!'

        for item in fvalidateAcc:
            validateAcc.append({'x':item['x']+epoch,'acc':item['acc']})
            if item['x']+epoch>maxEpoch:
                maxEpoch=item['x']+epoch
        print 'validateAcc loaded!'

        for item in ftestAcc:
            testAcc.append({'x':item['x']+epoch,'acc':item['acc']})
            if item['x']+epoch>maxEpoch:
                maxEpoch=item['x']+epoch
        print 'testAcc loaded!'

        epoch=maxEpoch

    cPickle.dump([result,trainAcc,validateAcc,testAcc,costValue],open(outputfile,'wb'))
