#!/usr/bin/env python
# coding: utf-8
#https://github.com/adambielski/siamese-triplet
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


seed = 14
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms= True

# This is for the Contrastive models
class EmbeddingNet(nn.Module):

    def __init__(self, arch, n=None, trained=False):
        super(EmbeddingNet, self).__init__()
        # Get resnet model, potentially pretrained
        self.resnet = arch(pretrained=trained)
    
        # Over-write the first conv layer to be able to read gray scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # Remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Add linear layers to compare between the features of the two images notice the two dimensional output
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
        # Apply layer freezing if applicable
        length = len(list(self.resnet.parameters()))
        if n != None:
            for param in list(self.resnet.parameters())[:int(-n*length/64)]:
                param.requires_grad = False

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.resnet(x)
        # flattens
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def get_embedding(self, x):
        return self.forward(x)

# Shared Contrastive network
class sharedContrastive(nn.Module):
    def __init__(self, embedding_net):
        super(sharedContrastive, self).__init__()
        # Has a single embedded network 
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        # We run the inputs through the same network
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
    
# Contrastive network when unshared    
class unsharedContrastive(nn.Module):
    def __init__(self, embedding_net, embedding_net2):
        super(unsharedContrastive, self).__init__()
        # We have two seperate networks
        self.embedding_net = embedding_net
        self.embedding_net2 = embedding_net2
        

    def forward(self, x1, x2):
        # The forward process runs them through the seperate networks
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net2(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

# Unshared aggregated Siamese network   
class unsharedAggregated(nn.Module):

    def __init__(self, arch, n=None, trained=False):
        
        def squeeze_weights(m):
            m.weight.data = m.weight.data.sum(dim=1)[:,None]
            m.in_channels = 1
            
        super(unsharedAggregated, self).__init__()
        
        # Gets the architecture, can be pretrained
        self.resnet = arch(pretrained = trained)
        self.resnet2 = arch(pretrained = trained) 
        
        # Over-write the first conv layer to be able to read gray scale channel
        # Works for different architectures 
        if arch.__name__ == "resnet18":
            self.resnet2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc_in_features = self.resnet.fc.in_features
        
        elif arch.__name__ == "alexnet":
            #
            self.resnet.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet2.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc_in_features = self.resnet.classifier[1].in_features
            
        elif arch.__name__ == "efficientnet_v2_s":
            self.resnet.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.resnet2.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.fc_in_features = self.resnet.classifier[1].in_features   
            
        elif arch.__name__ == "mobilenet_v3_small":
            #
            self.resnet.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.resnet2.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.fc_in_features = self.resnet.classifier[0].in_features  
            
        elif arch.__name__ == "vgg11":
            #
            self.resnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.resnet2.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.fc_in_features = self.resnet.classifier[0].in_features  
            
        elif arch.__name__ == "densenet121":
            # weird because is off by 81 = 3^4
            self.resnet.features.conv0.apply(squeeze_weights)
            self.resnet2.features.conv0.apply(squeeze_weights)
            self.fc_in_features = self.resnet.classifier.in_features * 81
        
        elif arch.__name__ == "vit_b_16":
            self.resnet.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            self.resnet2.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            self.fc_in_features = list(self.resnet.children())[-1][-1].in_features
        
        # Remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.resnet2 = torch.nn.Sequential(*(list(self.resnet2.children())[:-1]))
        
        # Add linear layers to compare between the features of the two images and produce a single output
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        
        # Freeze layers if applicable
        length = len(list(self.resnet.parameters()))
        if n != None:
            for param,param2 in zip(list(self.resnet.parameters())[:int(-n*length/64)],list(self.resnet2.parameters())[:int(-n*length/64)]):
                param.requires_grad = False
                param2.requires_grad = False
        
        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.resnet2.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output
    
    def forward_once2(self, x):
        output = self.resnet2(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # Gets two images' features, we have two seperate forward processes
        output1 = self.forward_once(input1)
        output2 = self.forward_once2(input2)

        # After the subnetworks have done the foward process we concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # Passes the concatenation to the linear layers (classifier)
        output = self.fc(output)

        # Pass the output of the linear layers to sigmoid layer which rescales to (0,1) for class labels
        output = self.sigmoid(output)
        if len(output)==1:
            return output[0]
        return output.squeeze()
    
# Shared aggregated Siamese network    
class sharedAggregated(nn.Module):

    def __init__(self, arch, n=None, trained=False):
        super(sharedAggregated, self).__init__()
        # get resnet model
        self.resnet = arch(pretrained = trained)
        # over-write the first conv layer to be able to read gray scale channel
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        length = len(list(self.resnet.parameters()))
        if n != None:
            for param in list(self.resnet.parameters())[:int(-n*length/64)]:
                param.requires_grad = False
            
        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # This time we have a single forward process through the same network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        if len(output)==1:
            return output[0]
        return output.squeeze()
    
# Baseline network    
class baseline(nn.Module):

    def __init__(self, arch, n=None, trained=False):
        super(baseline, self).__init__()
        # get resnet model
        self.resnet = arch(pretrained = trained)

        # over-write the first conv layer to be able to read gray scale channel
        if arch.__name__ == "resnet18" or "resnet50":
        # over-write the first conv layer to be able to read gray scale channel
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc_in_features = self.resnet.fc.in_features
            
        elif arch.__name__ == "vgg11":
            self.resnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.fc_in_features = self.resnet.classifier[0].in_features
            
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        length = len(list(self.resnet.parameters()))
        if n != None:
            for param in list(self.resnet.parameters())[:int(-n*length/64)]:
                param.requires_grad = False

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1):
        # Only one forward input here into a single network
        output = self.forward_once(input1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        if len(output)==1:
            return output[0]
        return output.squeeze()

