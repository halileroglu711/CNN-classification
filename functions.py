import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
device=("cuda"if torch.cuda.is_available()else "cpu")
label_dict={0:"Airplane",1:"Car",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",7:"Horse",8:"Ship",9:"Truck"}

#Get train and test loaders
def get_data_loaders(batch_size=64):
    transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#converts image to tensor 
        transforms.Normalize(((0.5,0.5,0.5)),(0.5,0.5,0.5))#works only with tensors and normalizes it's pixels between dev(average) and std each number stands for rgb channels in order
    ])
    train_set=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
    test_set=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader

#In order to visualize samples from train_loader, we are going to undo image normalization we applied before visualization.
#because we are working with RGB images in this project and RGB images looks different when they are normalized.
#we have these 3 functions for image visualization
def imshow(img):
    img=img / 2 + 0.5 #first reverse normalization 
    np_img=img.numpy()#converts image to numpy again
    plt.imshow(np.transpose(np_img,(1,2,0))) #shows colors correctly for 3 channels (in order.)
def get_sample_images(train_loader): #gets images from train_loader
    data_iter=iter(train_loader)
    images,labels=next(data_iter)
    return images,labels
def visualize(n,train_loader): #finally visualizes n images
    images,labels=get_sample_images(train_loader)
    plt.figure()
    plt.title("A few data samples from training dataset")
    plt.axis("off")
    for i in range(n): #n iterations for image count
        plt.subplot(1,n,i+1)
        imshow(images[i])#visualize
        plt.title(f"{label_dict[labels[i].item()]}")
        plt.axis("off")
    plt.savefig("assets/random-sample-images.png")
    plt.show()

#Loss function and optimizer
define_loss_and_optimizer= lambda model:(
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)#stochastic gradient descent. 
    #SGD does not forget the weight's previous gradient and uses it to update the weight again if the next gradiant goes to same direction.
)

#Train
def train_model(model,n,train_loader,criterion,optimizer,scheduler,epochs=5):
    model.train()
    train_losses=[]
    visualize(n,train_loader)
    for epoch in range(epochs):
        total_loss=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            
            optimizer.zero_grad() #refresh gradiants
            
            predictions=model(images)#forward propagation (get predictions)
            loss=criterion(predictions,labels)#compute loss
            loss.backward()#backward propagation, calculate gradients
            optimizer.step()#LEARNING = update weights
            
            total_loss+=loss.item()
        scheduler.step() #Decrease LR progressively after each epoch
        avg_loss=total_loss/len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs} Loss: {avg_loss}")
        
    torch.save(model.state_dict(),"weights/best.pt")
    print("Training process has finished. Final weights were saved as 'weights/best.pt' ")
    #Visualize train_losses in time.
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label="Train Loss")
    plt.xlabel("Epochs")        
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("assets/training-loss-graph.png")
    plt.show()

#Test
def test_model(model,test_loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            predictions=model(images)
            _,predicted=torch.max(predictions,1)#Get the highest value among the 10 predictions
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    accuracy= f"{100*correct/total:.3f}"
    return accuracy