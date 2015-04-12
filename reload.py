import os
import sys

if __name__=='__main__':
	if len(sys.argv)!=3:
		print 'Usage: python reload.py <General Folder> <Model to Reload>'
		exit(0)

	if os.name!='posix':
		print 'Unix/Linux Only'
		exit(0)

	folder=sys.argv[1]
	model=sys.argv[2]
	if folder[-1]=='/':
		folder=folder[:-1]

	if 'tmp' in os.listdir('.'):
		os.system('rm -rf tmp')

	os.system('mkdir tmp')
	os.system('cp -r %s/* tmp/'%folder)

	with open(model,'r') as fopen:
		command=fopen.readline()
		currentFile=''
		data=''
		for line in fopen:
			if line[:4]=='####':
				if currentFile!='':
					with open('tmp/%s'%currentFile,'w') as fwrite:
						fwrite.write(data)
					data=''
				pt1=0;pt2=len(line)
				state=0
				for i in xrange(len(line)):
					if state==0 and line[i]!='#':
						state=1
						pt1=i
					elif state==1 and line[i]=='#':
						pt2=i
						break
				currentFile=line[pt1:pt2]
			else:
				data+=line
		with open('tmp/%s'%currentFile,'w') as fwrite:
			fwrite.write(data)
		os.chdir('tmp')
		print command
		os.system(command)

	print 'reload completed!'
