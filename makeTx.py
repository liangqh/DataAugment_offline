import os
import random

trainval_percent = 0
train_percent = 1
xmlfilepath = 'data/Annotations_test'
txtsavepath = 'data/ImageSets_test'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

#ftrainval = open('data/ImageSets_train/trainval.txt', 'w')
ftrain = open('data/ImageSets_test/test.txt', 'w')
#fval = open('data/ImageSets_train/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    print(name)
    '''
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            pass
            #ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
    '''
    if i in trainval:
        pass
    else:
        ftrain.write(name)

print(i)
#ftrainval.close()
ftrain.close()
#fval.close()
#ftest.close()
