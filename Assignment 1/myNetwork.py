import torch
import torch.nn as nn
from RainforestDataset import get_classes_list


class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # Select all parts of the two pretrained networks, except for
        # the last linear layer.
        # from here: https://discuss.pytorch.org/t/how-to-concatenate-layers-in-pytorch-similar-to-tf-keras-layers-concatenate/33736/6
        self.fully_conv1 = nn.Sequential(*(list(pretrained_net1.children())[:-1]))
        self.fully_conv2 = nn.Sequential(*(list(pretrained_net2.children())[:-1]))
        
        # Create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        #print(self.fully_conv1.fc.in_features + self.fully_conv2.fc.in_features)
        inLen = pretrained_net1.fc.in_features + pretrained_net2.fc.in_features
        self.linear = nn.Linear(inLen, num_classes)


    def forward(self, inputs1, inputs2):
        # Feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.
        out1 = self.fully_conv1(inputs1)
        out2 = self.fully_conv2(inputs2)
        outAll = torch.cat((out1, out2), 1)
        outAll = outAll.view(outAll.size(0), -1)
        output = self.linear(outAll)
        return output


class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init=None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()


        if weight_init is not None:
            # Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.

            layer = pretrained_net.conv1
            weights = layer.weight
            s1,s2,s3,s4 = layer.weight.shape
            current_weights = torch.empty([s1,1,s3,s4])

            if weight_init == "kaiminghe":
                init = nn.init.kaiming_normal_(current_weights)
            else:
                init = nn.init.normal_(current_weights)
            new_weight = torch.cat((weights, init), 1)

            # Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)
            newConv = torch.nn.Conv2d(in_channels=4, out_channels=layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, bias=layer.bias)
            newConv.weight = nn.Parameter(new_weight)
            pretrained_net.conv1 = newConv


        # Overwrite the last linear layer.
        num_ftrs = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(num_ftrs, num_classes)
        #pretrained_net.fc = nn.Linear(512, num_classes)

        self.net = pretrained_net

    def forward(self, inputs):
        return self.net(inputs)





