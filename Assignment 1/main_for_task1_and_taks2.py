import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch import Tensor

import time
from datetime import datetime
import os
import numpy as np
import math

import PIL.Image
import sklearn.metrics

from typing import Callable, Optional

from RainforestDataset import RainforestDataset, ChannelSelect
from myNetwork import SingleNetwork, TwoNetworks

from RainforestDataset import get_classes_list

torch.backends.cudnn.deterministic = True 


def train_epoch(model, trainloader, criterion, device, optimizer):
  model.train()
  
  losses = []
  for batch_idx, data in enumerate(trainloader):
      if (batch_idx %100==0) and (batch_idx>=100):
        print('at batchidx',batch_idx)

      # Calculate the loss from your minibatch.
      # If you are using the TwoNetworks class you will need to copy the infrared
      # channel before feeding it into your model. 
      inputs = data['image'].to(device)        
      labels = data['label'].to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs,labels.to(device).float())

      loss.backward()
      optimizer.step()
      losses.append(loss.item())
  return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):
  model.eval()

  #curcount = 0
  #accuracy = 0 
  
  concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
  concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
  avgprecs=np.zeros(numcl) #average precision for each class
  fnames = [] #filenames as they come out of the dataloader

  with torch.no_grad():
    losses = []
    for batch_idx, data in enumerate(dataloader):
        if (batch_idx%100==0) and (batch_idx>=100):
          print('at val batchindex: ', batch_idx)
    
        inputs = data['image'].to(device)        
        outputs = model(inputs)
  
        labels = data['label'].to(device)

        loss = criterion(outputs, labels.to(device).float())
        losses.append(loss.item())

        # This was an accuracy computation
        # cpuout= outputs.to('cpu')
        # _, preds = torch.max(cpuout, 1)
        # labels = labels.float()
        # corrects = torch.sum(preds == labels.data)
        # accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )
        # curcount+= labels.shape[0]
        
        # Collect scores, labels, filenames
        predictions_out = outputs.to('cpu') # same as output before
        concat_pred = np.concatenate((concat_pred, predictions_out), axis=0)
        labels_out = labels.to('cpu')
        concat_labels = np.concatenate((concat_labels, labels_out), axis=0)
        fnames.extend(data['filename']) # extend because each data have "batch size" number of filename
  for c in range(numcl): 
    avgprecs[c]= sklearn.metrics.average_precision_score(concat_labels[:,c],concat_pred[:,c])
    
  return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test,  model,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  
  best_concat_labels = None
  best_concat_pred = None

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss = train_epoch(model,  dataloader_train,  criterion,  device, optimizer)
    print(avgloss)
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()
    
    perfmeasure, testloss, concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(np.nan_to_num(perfmeasure))
    # avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)



    if avgperfmeasure > best_measure:
      # Track current best performance measure and epoch
      bestweights = model.state_dict()
      best_measure = avgperfmeasure
      best_epoch = epoch

      # save best concats
      best_concat_labels = concat_labels
      best_concat_pred = concat_pred
      # save AP everytime when we found better avarage measures
      f = open("./output/modelBestPerEpoch.txt", "a")
      f.write(f"Epoch {epoch} ({avgperfmeasure}): {perfmeasure.tolist()}\n")
      f.close()

      # save/override best model, because I only save perfmeasure when I found better avgperfmeasure.
      torch.save(model.state_dict(), "./output/model_state_dict.pth") # this one is enough to load it and evaluate
      #torch.save(model, f"model_seed.pth")
  
  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, perfmeasure, best_concat_labels, best_concat_pred, fnames


# class yourloss(nn.modules.loss._Loss):

#     def __init__(self, reduction: str = 'mean') -> None:
#         #TODO
#         pass
    
#     def forward(self, input_: Tensor, target: Tensor) -> Tensor:
#         #TODO
#         pass
#         return loss

def plotTestTrain():
  maxnumepochs = 35 # might need to change it to more flexibal solution than hardcoding
  trainlosses = np.load('./output/trainlosses.npy')
  testlosses = np.load('./output/testlosses.npy')
  testperfs = np.load('./output/testperfs.npy')
  x = np.linspace(1,maxnumepochs,maxnumepochs)
  plt.plot(x, trainlosses, label='Train loss')
  plt.plot(x, testlosses, label='Test loss')  
  plt.plot(x,np.mean(testperfs, axis=1), label='mean AP')

  plt.legend()
  plt.xlabel("Epoch number")
  #plt.show()
  plt.savefig('./output/train_test_loss_and_testmAP')

def calculateTailAcc(seed):
  # calculate Tailacc
  # --------------------------------------------------
  concat_labels = np.load(f"./output/concat_labels.npy")
  concat_pred = np.load(f"./output/concat_preds.npy")
  fnames = np.load(f"./output/fnames.npy", allow_pickle = True)
  perfmeasure = np.load(f"./output/perfmeasure.npy")
  # --------------------------------------------------
  classList, num_classes = get_classes_list()
  classesAP = perfmeasure.tolist()
  max_value = max(classesAP)
  max_index = classesAP.index(max_value)

  zipped = list(zip(concat_labels, concat_pred, fnames))
  taccEachClass = {}
  # maxT = 0
  # for predList in preds:
  #   maxValue = max(predList)
  #   if maxValue > maxT:
  #     maxT = maxValue
  # print(maxT) # what is maxT for us?

  def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

  tvalues = np.linspace(0.1, 1, 20)
  for classIndex in range(num_classes):
    tmpList = []
    for t in tvalues:
      firstPart = 0
      secondPart = 0
      for i in range(len(zipped)):
        if sigmoid(zipped[i][1][classIndex]) > t:
          firstPart += 1
          # check if true value for this class i 1
          if zipped[i][0][classIndex] == 1:
            secondPart += 1
      if firstPart != 0: # so we won't devide on 0
        tmpList.append((1/firstPart)*secondPart)
      else:
        tmpList.append(1)
    taccEachClass[classList[classIndex]] = tmpList

  # for key in taccEachClass:
  #   # this part is needed for testing with lower number of images:
  #   if len(taccEachClass[key]) > 0:
  #     plt.plot(tvalues, taccEachClass[key], label=f"{key}")
  
  yList = np.zeros(len(tvalues)) # avarage of all AP 
  for key in taccEachClass:
    # this part is needed for testing with lower number of images:
    yList += np.array(taccEachClass[key])

  yList /= len(taccEachClass)

  plt.figure()
  plt.plot(tvalues, yList, label="avg of 17 classes")
  plt.xlabel("t value")
  plt.ylabel("Tailacc(t)")
  plt.legend()
  #plt.show()
  plt.savefig(f"./output/TailAcc_seed{seed}")




def runstuff(seed, imageTIFDir, csvPath, imageJPGDir=None):
  config = dict()
  config['use_gpu'] = True
  #config['use_gpu'] = False #True #TODO change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 32
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  torch.manual_seed(seed) # seed

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(channels=[0, 1, 2]),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(channels=[0, 1, 2]),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  # [0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137] ???


  # Datasets
  image_datasets={}
  image_datasets['train']=RainforestDataset(root_dir=imageTIFDir, csvPath = csvPath, trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=RainforestDataset(root_dir=imageTIFDir, csvPath = csvPath, trvaltest=1, transform=data_transforms['val'])
  
  # Dataloaders
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], 
                                    batch_size = config['batchsize_train'],
                                    num_workers=1) # faster with 4
  dataloaders['val'] = DataLoader(image_datasets['val'], 
                                  batch_size = config['batchsize_val'],
                                  num_workers=1)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model
  # Create an instance of the network that you want to use.
  #model = # TwoNetworks()
  # https://www.programcreek.com/python/example/108007/torchvision.models.resnet18
  pre_trained_model = models.resnet18(pretrained = True)
  model = SingleNetwork(pre_trained_model)

  model = model.to(device)

  # not sure if we can just use import for that or we had to write own loss function for that. 
  lossfct = nn.BCEWithLogitsLoss()
  #lossfct = yourloss()
  
  # Observe that all parameters are being optimized
  # Because this post:
  # https://www.reddit.com/r/deeplearning/comments/8hmy2l/what_is_the_best_tesorflow_optimizer_for_multi/
  # I decided to use Adam optimizer for multi-label classification
  optimizer = optim.Adam(params = model.parameters(), lr = config['lr']) # optimize on whole network

  # Decay LR by a factor of 0.3 every X epochs
  # looks like StepLR is what we need to decay LR by a factor of 0.3 every X epochs
  lrscheduler = lr_scheduler.StepLR(optimizer, step_size = config['scheduler_stepsize'], gamma = config['scheduler_factor'])
  
  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, perfmeasure, concat_labels, concat_pred, fnames = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model,  lossfct, optimizer, lrscheduler, num_epochs = config['maxnumepochs'], device = device, numcl = config['numcl'])

  # save last model
  torch.save(model.state_dict(), "./output/model_last.pth")


  np.save('./output/best_epoch', best_epoch)
  np.save('./output/best_measure', best_measure)
  np.save('./output/trainlosses', trainlosses)
  np.save('./output/testlosses', testlosses)
  np.save('./output/testperfs', testperfs) # test mAP

  # save best score
  np.save('./output/concat_labels', concat_labels)
  np.save('./output/concat_preds', concat_pred)
  np.save('./output/fnames', fnames)
  np.save('./output/perfmeasure', perfmeasure)
  

  # find top images and save it in output/images
  classesAP = perfmeasure.tolist()
  max_value = max(classesAP)
  max_index = classesAP.index(max_value)

  zipped = list(zip(concat_labels, concat_pred, fnames))
  def myFunc(e):
    if e[0][max_index] == 1: # check class
      return e[1][max_index] # return AP of this class
    else:
      return 0 # if class is not presented in the photo

  zipped.sort(reverse=True, key=myFunc)

  if imageJPGDir is not None:
    classList, num_classes = get_classes_list()

    # save filenames to the file and save top 10 images in the file
    f = open(f"./output/topImages_class{max_index}_seed{seed}.txt", "w")
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Top 10 for '{classList[max_index]}' class", fontsize=12)
    subPlotPosition = 1
    for i in range(50):
      fileName = zipped[i][2]
      f.write(f"{i+1}. {fileName}\n")

      if i < 10: # print 10 best
        img = PIL.Image.open(f"{imageJPGDir}/{fileName}.jpg")
        subP = fig.add_subplot(3, 4, subPlotPosition)
        subPlotPosition += 1
        subP.title.set_text(fileName)
        subP.axis('off')
        plt.imshow(img)
    
    #plt.show() 
    plt.savefig(f"./output/Top10_seed{seed}")
    f.close()

    # find 10 worse for best class
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Worse 10 for '{classList[max_index]}' class", fontsize=12)
    subPlotPosition = 1
    for imgZip in zipped[-11:-1]: # ten last in sorted list
      fileName = imgZip[2]
      img = PIL.Image.open(f"{imageJPGDir}/{fileName}.jpg")
      subP = fig.add_subplot(3, 4, subPlotPosition)
      subPlotPosition += 1
      subP.title.set_text(fileName)
      subP.axis('off')
      plt.imshow(img)
    plt.savefig(f"./output/Worse10_seed{seed}")
  
if __name__=='__main__':
  pathToDataSet = './' # This one you can change and define where is your dataset

  pathToCSV = pathToDataSet + 'train_v2.csv'# Here I suppose that csv file is in the dataSetFolder
  pathToTIF = pathToDataSet + 'train-tif-v2'
  pathToJPG = './train-jpg' # Path to JPG images. Change it if needed, or set it to None because main function needs it
  seed = 10
  runstuff(seed, pathToTIF, pathToCSV, imageJPGDir = pathToJPG)
  #calculateTailAcc(seed)
  #plotTestTrain()