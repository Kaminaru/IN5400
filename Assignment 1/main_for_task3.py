
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
      inputsFrom1 = data['image'][:,[0,1,2], : , :].to(device)
      inputsFrom2 = data['image'][:,[3], : , :].repeat(1,3,1,1).to(device)

      labels = data['label'].to(device)

      optimizer.zero_grad()
      outputs = model(inputsFrom1, inputsFrom2)
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

        inputsFrom1 = data['image'][:,[0,1,2], : , :].to(device)
        inputsFrom2 = data['image'][:,[3], : , :].repeat(1,3,1,1).to(device)
        outputs = model(inputsFrom1, inputsFrom2)

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

        # TODO: collect scores, labels, filenames
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

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, perfmeasure, best_concat_labels, best_concat_pred, fnames

def runstuff(seed, imageTIFDir, csvPath, imageJPGDir):
  config = dict()
  config['use_gpu'] = True #TODO change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 32
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 10 # 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  torch.manual_seed(seed) # seed

  # Data augmentations.
  # [0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284] # RGBa
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
  }


  # Datasets
  image_datasets={}
  image_datasets['train']=RainforestDataset(root_dir=imageTIFDir, csvPath = csvPath, trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=RainforestDataset(root_dir=imageTIFDir, csvPath = csvPath, trvaltest=1, transform=data_transforms['val'])

  # Dataloaders
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'],
                                    batch_size = config['batchsize_train'],
                                    num_workers=2) # faster with 4
  dataloaders['val'] = DataLoader(image_datasets['val'],
                                  batch_size = config['batchsize_val'],
                                  num_workers=2)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model
  # Create an instance of the network that you want to use.
  pre_trained_model1 = models.resnet18(pretrained = True)
  pre_trained_model2 = models.resnet18(pretrained = True)
  model = TwoNetworks(pre_trained_model1, pre_trained_model2)

  model = model.to(device)
  lossfct = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(params = model.parameters(), lr = config['lr']) # optimize on whole network
  lrscheduler = lr_scheduler.StepLR(optimizer, step_size = config['scheduler_stepsize'], gamma = config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, perfmeasure, concat_labels, concat_pred, fnames = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model,  lossfct, optimizer, lrscheduler, num_epochs = config['maxnumepochs'], device = device, numcl = config['numcl'])

  # save model
  torch.save(model.state_dict(), os.path.join("./output/", f"model2pre_seed{seed}.pth"))


if __name__=='__main__':
  pathToDataSet = './' # This one you can change and define where is your dataset

  pathToCSV = pathToDataSet + 'train_v2.csv'# Here I suppose that csv file is in the dataSetFolder
  pathToTIF = pathToDataSet + 'train-tif-v2'
  seed = 10
  runstuff(seed, pathToTIF, pathToCSV)
