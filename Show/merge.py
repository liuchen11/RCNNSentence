import sys

begin='''
<!DOCTYPE HTML>
<html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Result</title>

    <script type="text/javascript"
    src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
    <style type="text/css">
    ${demo.css}
</style>
'''

end1='''
	</head>
	<body>
<script src="../js/highcharts.js"></script>
<script src="../js/modules/exporting.js"></script>
'''

end2='''
	</body>
</html>
'''

if __name__=='__main__':
	if len(sys.argv)<3:
		print 'Usage: python merge.py <src file >... <dest file>'
		exit(0)
		
	fileNum=len(sys.argv)-2
	fileName=sys.argv[-1]

	fwrite=open('./charts/'+fileName,'w')
	fwrite.write(begin)

	chartsNum=0
	for  htm in sys.argv[1:-1]:
		with open(htm,'r') as fopen:
			while True:
				line=fopen.readline()
				if line=='':
					break
				if line.find('<script type="text/javascript">')>=0:
					fwrite.write(line)
					line=fopen.readline()
					fwrite.write(line)
					fopen.readline()
					fwrite.write("    $('#container%i').highcharts({\n"%chartsNum)
					chartsNum+=1
					while True:
						line=fopen.readline()
						fwrite.write(line)
						if line.find('script')>=0:
							break

	fwrite.write(end1)
	for i in xrange(chartsNum):
		fwrite.write('<div id="container%i" style="min-width: 310px; height: 800px; margin: 0 auto"></div>\n'%i)
	fwrite.write(end2)
