inputFilePath='/usr/prakt/w065/stairs/dataset_test.txt'
outputFilePath = '/usr/prakt/w065/DLinCV/scripts/lstm-keras-tf/orderedSets/stairsorderedset.txt'

data = open(inputFilePath)
data.readline()
data.readline()
data.readline()
lines= data.readlines()
lines = sorted(lines)
data.close()
f = open(outputFilePath, 'w')
for l in lines:
    #if "color" in l:
        f.write("%s\n" % l)
f.close()  
