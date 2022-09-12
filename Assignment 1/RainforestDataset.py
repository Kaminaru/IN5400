import torch
from torch.utils.data import Dataset
import os
import PIL.Image
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, csvPath, trvaltest, transform):
        self.root_dir = root_dir
        # TODO: change it to the normal size image dictionary
        self.image_folder = root_dir  # 1000images
        self.label_file = csvPath # train_v2_1000.csv
        self.transform = transform
        
        df = pd.read_csv(self.label_file)
        labels = df['tags'].str.split(" ").tolist() # split all words and makes a list and save it
        filenames = df['image_name'].tolist()

        self.classes, self.num_classes = get_classes_list()
        # Binarise your multi-labels from the string
        multiLabelBZ = MultiLabelBinarizer()
        multiLabelBZ.fit([self.classes])
        labelsBinary = multiLabelBZ.transform(labels)
        # Perform a test train split. It's recommended to use sklearn's train_test_split with the following
        # parameters: test_size=0.33 and random_state=0 - since these were the parameters used
        # when calculating the image statistics you are using for data normalisation.
    
        # for debugging you can use a test_size=0.66 - this trains then faster
        # OR optionally you could do the test train split of your filenames and labels once, save them, and
        # from then onwards just load them from file.
        #train_test_split(filenames, labels, test_size=0.33, random_state=0)   # 0.66 faster
        train_img, test_img, train_label, test_label = train_test_split(filenames, labelsBinary, test_size=0.33, random_state=50)
        if trvaltest == 0: # training
            self.img_filenames = train_img
            self.img_labels = train_label
        elif trvaltest == 1: # validation
            self.img_filenames = test_img
            self.img_labels = test_label


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Get the label and filename, and load the image from file.
        format = ".tif"
        path_to_image = os.path.join(self.image_folder,self.img_filenames[idx]+format)
        img = PIL.Image.open(path_to_image)
        #img.show()
        if self.transform is not None: 
            img = self.transform(img)

        sample = {'image': img,
                  'label': self.img_labels[idx],
                  'filename': self.img_filenames[idx]}
        return sample