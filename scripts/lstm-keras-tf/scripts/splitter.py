import sys,getopt,os
def splitSet(outputDir,inputSet):
    prevSeq = ''
    outF=None
    with open(inputSet) as f:
            print 'preparing data'
            for line in f:
                if line.isspace():
                    continue
		if line.startswith('/'):
			line=line[1:]
                currentSeq = line.split('/')[0]
                if currentSeq !=prevSeq:
                    if outF is not None:
                        outF.close()
                    prevSeq = currentSeq
                    outF = open(os.path.join(outputDir,currentSeq+'.txt'), "a")

		print(line)
                outF.write(line+'\n')
            outF.close()


def main(argv):
   outputDir = ''
   inputSet = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["outputDir=","inputSet="])
      if len(opts)<2:
	print('splitter.py -o <outputDir> -i <inputSet>')
	sys.exit(2)
   except getopt.GetoptError:
      print ('splitter.py -o <outputDir> -i <inputSet>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('splitter.py -o <outputDir> -i <inputSet>')
         sys.exit()
      elif opt in ("-o", "--outputDir"):
         outputDir = arg
      elif opt in ("-i", "--inputSet"):
         inputSet = arg
   if not os.path.exists(outputDir):
    os.makedirs(outputDir)
   splitSet(outputDir,inputSet)


if __name__ == "__main__":
   main(sys.argv[1:])
