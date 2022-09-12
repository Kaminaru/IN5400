from utils.dataLoader import DataLoaderWrapper
from utils.saverRestorer import SaverRestorer
from utils.model import Model
from utils.trainer import Trainer
from utils.validate import plotImagesAndCaptions
from utils.validate_metrics import validateCaptions
from cocoSource_xcnnfused import ImageCaptionModel

import numpy as np
import torch
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True 

def main(config, modelParam):
    # create an instance of the model you want
    model = Model(config, modelParam, ImageCaptionModel)

    # create an instacne of the saver and resoterer class
    saveRestorer = SaverRestorer(config, modelParam)

    model = saveRestorer.restore(model) # load model

    # create your data generator
    dataLoader = DataLoaderWrapper(config, modelParam)

    #plotImagesAndCaptions
    plotImagesAndCaptions(model, modelParam, config, dataLoader)
    validateCaptions(model, modelParam, config, dataLoader)

    # Print the loss:
    trainObj = Trainer(model, modelParam, config, dataLoader, saveRestorer)
    epoch_loss = trainObj.run_epoch('val', model, True, 'best')
    print(f'Best epoch loss: {epoch_loss}') 


########################################################################################################################
if __name__ == '__main__':
    data_dir = '/itf-fi-ml/shared/IN5400/dataforall/mandatory2/data/coco/'

    #train
    modelParam = {
        'batch_size': 128,  # Training batch size
        'cuda': {'use_cuda': True,  # Use_cuda=True: use GPU
                 'device_idx': 0},  # Select gpu index: 0,1,2,3
        'numbOfCPUThreadsUsed': 10,  # Number of cpu threads use in the dataloader
        'numbOfEpochs': 99,# 99  # Number of epochs
        'data_dir': data_dir,  # data directory
        'img_dir': 'loss_images_test/',
        'modelsDir': 'storedModels_test/',
        'modelName': 'model_0/',  # name of your trained model    # change name run another task
        'restoreModelLast': 0,
        'restoreModelBest': 0,
        'modeSetups': [['train', True], ['val', True]],
        'inNotebook': False,  # If running script in jupyter notebook
        #'inference': True # restore model
    }

    config = {
        'optimizer': 'adamW',  # 'SGD' | 'adam' | 'RMSprop' | 'adamW' 
        'learningRate': {'lr': 0.001},  # learning rate to the optimizer
        'weight_decay': 0.00001,  # weight_decay value
        'number_of_cnn_features': 2048,  # Fixed, do not change
        'embedding_size': 300,  # word embedding_layer size
        'vocabulary_size': 10000,  # number of different words
        'truncated_backprop_length': 25,
        'hidden_state_sizes': 512,  #
        'num_rnn_layers': 1,  # number of stacked rnn's
        'scheduler_milestones': [75,90], #45,70 end at 80? or 60, 80
        'scheduler_factor': 0.2, #+0.25 dropout
        #'featurepathstub': 'detectron2vg_features' ,
        #'featurepathstub': 'detectron2m_features' ,
        #'featurepathstub': 'detectron2cocov3_tenmfeatures' ,
        'featurepathstub': 'detectron2_lim10maxfeatures' ,
        'cellType':  'RNN' #'GRU'  # RNN or GRU or GRU??
    }

    
    modelParam['batch_size'] = 64
    modelParam['modeSetups'] = [['val', False]]
    modelParam['restoreModelBest'] = 1

    main(config, modelParam)

    aa = 1
