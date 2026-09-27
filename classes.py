from functions import *
#Create CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)# First convolution layer
        self.bn1=nn.BatchNorm2d(32)#batch normalization
        self.relu=nn.ReLU()#activation function
        
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)#Pooling layer
        
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)#second convolution layer
        #which applies 64 different filters(they are looking for curves,lines,circles) to 32 feature maps
        
        self.bn2=nn.BatchNorm2d(64) # batch normalization for second convolution layer
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.bn4=nn.BatchNorm2d(256)
        
        self.fc1=nn.Linear(256*2*2,128)#first fully connected layer.
        
        self.dropout=nn.Dropout(0.2)#switches off the 20 percent of neurons during training
        
        self.fc2=nn.Linear(128,10)#second fully connected layer.
        
    def forward(self,x):
        x=(self.pool(self.relu(self.bn1(self.conv1(x))))) #first conv-bn-relu-pooling
        
        x=(self.pool(self.relu(self.bn2(self.conv2(x))))) #second conv-bn-relu-pooling
        
        x=(self.pool(self.relu(self.bn3(self.conv3(x))))) #third conv-bn-relu-pooling
        
        x=(self.pool(self.relu(self.bn4(self.conv4(x))))) #fourth conv-bn-relu-pooling
        
        x=torch.flatten(x,start_dim=1)#transforms feature maps into 1d vector = [256*2*2] , takes first index for starting
        
        x=self.dropout(self.relu(self.fc1(x)))#first fully connected layer 
        
        x=(self.fc2(x))#second fully connected layer
        return x #returns 10 different predictions (class count)