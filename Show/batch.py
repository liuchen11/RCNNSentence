import os
import sys
import re

if __name__=='__main__':
    if len(sys.argv)<2:
        print 'Usage: python batch.py <file pattern>'
        exit(0)

    for i in xrange(1,len(sys.argv)):
        command='python gen.py '+sys.argv[i]
        os.system(command)

#    pattern=sys.argv[1]
#    path='.'
#    if pattern.find('/')>=0:
#        pt=pattern.find('/')
#        while pt>=0:
#            split=pt
#            pt=pattern.find('/',pt+1)
#        path=pattern[:split]
#        pattern=pattern[split+1:]

#    files=os.listdir(path)
#    pattern=pattern.replace('*','.+')
#    for f in files:
#        if re.match(pattern,f):
#            command='python gen.py '+path+'/'+f
#            print command
#            os.system(command)
