import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch import Tensor
import numpy as np
import sklearn.metrics
from RainforestDataset import RainforestDataset, ChannelSelect
from myNetwork import SingleNetwork, TwoNetworks

torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    pathToDataSet = './' # This one you can change and define where is your dataset

    pathToCSV = pathToDataSet + 'train_v2.csv'
    pathToTIF = pathToDataSet + 'train-tif-v2'

    seed = 10
    torch.manual_seed(seed)
    pre_trained_model = models.resnet18(pretrained = True)
    model = SingleNetwork(pre_trained_model)
    device= torch.device('cuda:0')
    model = model.to(device)

    model.load_state_dict(torch.load('./output/model_state_dict.pth')) # load best model
    model.eval()



    data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect(channels=[0, 1, 2]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    image_datasets =RainforestDataset(root_dir=pathToTIF, csvPath = pathToCSV, trvaltest=1, transform=data_transforms['val'])

    # Dataloaders
    dataloader = DataLoader(image_datasets, batch_size = 64, num_workers=1)

    numcl = 17

    concat_pred = np.empty((0, numcl))
    concat_labels = np.empty((0, numcl))
    avgprecs=np.zeros(numcl)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = data['image'].to(device)        
            outputs = model(inputs)
            labels = data['label'].to(device)

            predictions_out = outputs.to('cpu')

            predictions_out = outputs.to('cpu') # same as output before
            concat_pred = np.concatenate((concat_pred, predictions_out), axis=0)
            labels_out = labels.to('cpu')
            concat_labels = np.concatenate((concat_labels, labels_out), axis=0)

    for c in range(numcl): 
        avgprecs[c]= sklearn.metrics.average_precision_score(concat_labels[:,c],concat_pred[:,c])

    concat_predFromFile = np.load(f"./output/concat_preds.npy") # best concat_preds

    sameList = 0
    lenL = len(concat_predFromFile)
    for i in range(lenL):
        if sum(concat_predFromFile[i]) == sum(concat_pred[i]):
            sameList += 1

    print(f"Identical {sameList} from {lenL}")

    sum1 = concat_predFromFile[0] # 17 classes
    sum2 = concat_pred[0]
    for i in range(1, len(concat_pred)):
        sum1 += concat_predFromFile[i]
        sum2 += concat_pred[i]
    
    print("Total difference between two concat_pred for each class:\n" , (sum1 - sum2)/lenL)
    print("Total difference between two concat_pred:", sum(sum1 - sum2)/lenL)