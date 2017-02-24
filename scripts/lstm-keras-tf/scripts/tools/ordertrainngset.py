pre='office'
s3t='test'


inputFilePath='/usr/prakt/w065/'+pre+'/dataset_'+s3t+'.txt'
outputFilePath = '/usr/prakt/w065/DLinCV/scripts/lstm-keras-tf/orderedSets/'+pre+'orderedset_'+s3t+'.txt'

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
