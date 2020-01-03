import numpy as np
from skimage import io
from skimage import transform
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from my_data import  MyDataset
from DenseNet import DenseNet
import csv

BATCH_SIZE = 18
NUM_EPOCHS = 25
degree = 45

def binary(c):
    for index in range(len(c)):
        if c[index]>0.5:
            c[index] = 1
        else:
             c[index] = 0
    return c


normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=degree, resample=False, expand=False, center=None),
	transforms.ToTensor(),
	normTransform
	])   

test_dataset = MyDataset(root_dir='./archive',
                         root_dir1='/test/',
                         names_file='/sampleSubmission.csv',
                         transform=transform)

test_loader = DataLoader(dataset = test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda')
model = DenseNet(growthRate=24,depth=15,reduction=0.5,nClasses=1,bottleneck=True).to(device)
model.load_state_dict(torch.load("DenseNet_parameter3.pkl"))
model.eval() #eval 模式


with open("E:/progaramming/python_files/archive_test.csv","w", newline = '') as csvfile: 
    writer = csv.writer(csvfile) 
    for i, data in enumerate(test_loader):
        X_test,Y_test = data
        X_test = np.reshape(X_test,[-1,1,32,32,32]).cuda()
        Y_test = Y_test.cuda()
        Y_test = Y_test.float()

        model.eval()
        outputs = model(X_test).cpu()
        # outputs = torch.softmax(outputs,1)
        outputs = outputs.detach().numpy()
        # print(outputs)
        writer.writerows(outputs)
    
csvfile.close()

csvFile = open("archive_test.csv", "r")
reader = csv.reader(csvFile)

result1 = {}
for item in reader:
    result1[reader.line_num] = item

csvFile.close()
# print(result1[1])

csvFile = open("./ykl_sub.csv")
reader1 = csv.reader(csvFile)
candidate = {}
for item in reader1:
    if reader1.line_num == 1:
        fileHeader = item
        continue
    candidate[reader1.line_num-1] = item

csvFile.close
# print(candidate[1])

for i in candidate:
    candidate[i][1] = result1[i][0]
# print(candidate[1])

csvFile = open("./ykl_sub_test.csv", "w",newline='')
writer = csv.writer(csvFile)
writer.writerow(fileHeader)
for i in candidate:
    writer.writerow(candidate[i])

csvFile.close()