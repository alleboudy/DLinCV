inputFilePath='chess_train.txt'
outputFilePath = 'CHorderedset.txt'

data = open(inputFilePath)
data.readline()
data.readline()
data.readline()
lines= data.readlines()
lines = sorted(lines)
data.close()
f = open(outputFilePath, 'w')
for l in lines:
    f.write("%s\n" % l)
f.close()  
