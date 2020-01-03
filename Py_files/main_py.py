import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from my_data import  MyDataset
from torchsummary import summary
from DenseNet import DenseNet
import csv
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from sqlalchemy.orm import evaluator

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
def get_batch(x, y, batch_size, alpha=0.2):
    """
    get batch data
    :param x: training data
    :param y: one-hot label
    :param step: step
    :param batch_size: batch size
    :param alpha: hyper-parameter α, default as 0.2
    :return:
    """
    x,y = x.cpu(), y.cpu()
    # offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)

    # get batch data
    x = x.detach().numpy()
    y = y.detach().numpy()
    n = len(x)

    # 最原始的训练方式
    if alpha == 0:
        return x, y
    # mixup增强后的训练方式
    if alpha > 0:
        lam = np.random.beta(alpha,alpha,n)
    indexs = np.random.randint(0,n,n)
    for i in range(n):
        x[i] = x[i]*lam[i]+(1-lam[i]*x[indexs[i]])
        y[i] = y[i]*lam[i]+(1-lam[i]*y[indexs[i]])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.cuda()
        y = y.cuda()
        return x, y

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

train_dataset = MyDataset(root_dir='./archive',
                          root_dir1='/train_val/',
                          names_file='/train_val.csv',
                          transform=transform)

valid_dataset = MyDataset(root_dir='./archive',
                          root_dir1='/train_val/',
                          names_file='/test_val.csv',
                          transform=transform)

test_dataset = MyDataset(root_dir='./archive',
                         root_dir1='/test/',
                         names_file='/sampleSubmission.csv',
                         transform=transform)

train_loader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(dataset = valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset = test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# 需要使用device来指定网络在GPU还是CPU运行
device = torch.device('cuda')
model = DenseNet(growthRate=24,depth=15,reduction=0.5,nClasses=1,bottleneck=True).to(device)
model.eval() #eval 模式

# TODO:define loss function and optimiter
criterion = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters())
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=10,verbose=False,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-08)

# train and evaluate
for epoch in range(NUM_EPOCHS):
    # lr_scheduler.step(epoch)
    running_loss = 0.0
    running_correct = 0.0

    print("Epoch {}/{}".format(epoch,NUM_EPOCHS))
    print("-"*10)
    for i,data in enumerate(train_loader):
    # TODO:forward + backward + optimize
        X_train,Y_train = data
        X_train = np.reshape(X_train,[-1,1,32,32,32]).cuda()
        Y_train = Y_train.float()
        Y_train = Y_train.cuda()
        
        X1_train,Y1_train = get_batch(X_train,Y_train,BATCH_SIZE,0.2)
        X_train = torch.cat((X_train,X1_train),0)
        Y_train = torch.cat((Y_train,Y1_train),0)

        optimizer.zero_grad()
        model.train()
        outputs = model(X_train)

        #print('train outputs output:{}'.format(outputs))
        #outputs = torch.softmax(outputs,1)

        loss = criterion(outputs,Y_train)
        loss.backward()
        optimizer.step()
        
        outputs = binary(outputs)
        Y_train = binary(Y_train)

        running_loss += loss.item()
        for index in range(len(outputs)):
            if outputs[index]==Y_train[index]:
                running_correct += 1
    
    # def score_function(engine):
    #     val_loss = engine.state.metrics['Accuracy']
    #     return val_loss
    
    # trainer = Engine(running_loss)
    # handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    # # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    # evaluator.add_event_handler(Events.COMPLETED, handler)            
    
    testing_correct = 0.0
    for i, data in enumerate(valid_loader):
        X_test,Y_test = data
        X_test = np.reshape(X_test,[-1,1,32,32,32]).cuda()
        Y_test = Y_test.cuda()
        Y_test = Y_test.float()

        model.eval()
        outputs = model(X_test)
        #outputs = torch.softmax(outputs,1)
        #print('test outputs output:{}'.format(outputs))

        outputs = binary(outputs)
        Y_test = binary(Y_test)
        #testing_correct +=torch.sum(outputs == Y_test.data)
        for index in range(len(outputs)):
            if outputs[index]==Y_test[index]:
                testing_correct += 1
    print("Loss is:{:.4f},Train Acurracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(running_loss/len(train_dataset),
                                                                                    100*running_correct/(2*len(train_dataset)),
                                                                                    100*testing_correct/len(valid_dataset))) 
    
    if 100*testing_correct/len(valid_dataset)>64:
        with open("E:/progaramming/python_files/archivetest1.csv","w", newline = '') as csvfile: 
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
                 torch.save(model.state_dict(),"DenseNet_parameter1.pkl")
    torch.save(model.state_dict(),"DenseNet_parameter.pkl")    
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset
with open("E:/progaramming/python_files/archivetest.csv","w", newline = '') as csvfile: 
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


