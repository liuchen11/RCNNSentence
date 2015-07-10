import os
import sys

if __name__=='__main__':
    if len(sys.argv)<2:
        print 'len of input arguments must be at list 2'
        exit(0)
    inputfiles=sys.argv[1:-1]
    outputfile=sys.argv[-1]

    print '%i input files: '%len(inputfiles),inputfiles
    print 'output file: ',outputfile

    for infile in inputfiles:
        dirNum=infile.count('/')
        sp=-1
        for i in xrange(dirNum):
            sp=infile.find('/',sp+1)
        sp0=infile.find('_',sp+1)
        sp1=infile.find('_',sp0+1)
        sp2=infile.find('_',sp1+1)
        sp3=infile.find('_',sp2+1)
        outputName=infile[sp+1:sp0]+'-'+infile[sp0+1:sp1]+'-'+infile[sp1+1:sp2]+'-'+infile[sp2+1:sp3]+'-'+outputfile
        command='THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python runDCNN.py -r %s -n %s -v ~/GoogleNews-vectors-negative300.bin -d ../SST1/data -static -word2vec -noval'%(infile,outputName)
        print command
        os.system(command)
    print 'Completed!'
