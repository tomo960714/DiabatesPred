import torch.nn as nn

class GenderNetwork(nn.Module):
    def __init__(self):
        super(GenderNetwork, self).__init__()

        #Temporal layer
        #8 convolutional layers
        #N filters, shaped as 1xK
        #K= 7,5,5,5,5,3,3,3
        #N = 16,16,32,32,64,64,64,64
        #MP factor =2,4,2,4,2,2,2
        
        self.conv1 = nn.Conv1d(1,16,7)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16,16,5)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(16,32,5)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(32,32,5)
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(32,64,5)
        self.batchnorm5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(64,64,3)
        self.batchnorm6 = nn.BatchNorm1d(64)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool1d(2)

        self.conv7 = nn.Conv1d(64,64,3)
        self.batchnorm7 = nn.BatchNorm1d(64)
        self.relu7 = nn.ReLU()
        self.maxpool7 = nn.MaxPool1d(2)

        self.conv8 = nn.Conv1d(64,64,3)
        self.batchnorm8 = nn.BatchNorm1d(64)
        self.relu8 = nn.ReLU()
        self.maxpool8 = nn.MaxPool1d(2)

        #Spatial filter
        #128 filters
        #shaped as 12x1
        #mp factor = 2

        self.conv9 = nn.Conv1d(64,128,12)
        self.batchnorm9 = nn.BatchNorm1d(128)
        self.relu9 = nn.ReLU()
        self.maxpool9 = nn.MaxPool1d(2)

        #Fully connected layer
        #(128,64)
        self.linear1 = nn.Linear(128,64)
        self.batchnorm10 = nn.BatchNorm1d(64)
        self.relu10 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        #(64,32)
        self.linear2 = nn.Linear(64,32)
        self.batchnorm11 = nn.BatchNorm1d(32)
        self.relu11 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        #softmax layer
        #(32,2)
        self.linear3 = nn.Linear(32,2)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        #Temporal filters
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        x = self.maxpool8(x)

        #Spatial filter
        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.relu9(x)
        x = self.maxpool9(x)

        #Fully connected layers
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.batchnorm10(x)
        x = self.relu10(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batchnorm11(x)
        x = self.relu11(x)
        x = self.dropout2(x)

        #Softmax layer
        x = self.linear3(x)
        x = self.softmax(x)

        return x
