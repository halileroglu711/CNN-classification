from classes import *
from functions import *
import torch.nn.functional as F
from PIL import Image
import cv2 as cv
#we need a new instance from CNN class in order to test the model.
t_model=CNN()

#Load the best weights from training process (best.pt)
t_model.load_state_dict(torch.load("weights/best.pt"))

#Image pre-process phase.
img=Image.open("assets/akinci.png").convert("RGB")
def getimg(img):
    transform=transforms.Compose([
        transforms.Resize((32,32)),#CIFAR-10 Default image shape
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    #Apply transforms
    img_tensor=transform(img)
    
    #Add batch dimension
    img_tensor=img_tensor.unsqueeze(0)
    
    return img_tensor

#Get the model's prediction.
def pred(img_tensor,img):
    t_model.eval()
    with torch.no_grad():
        output=t_model(img_tensor)#[1,10]
        poss=F.softmax(output,dim=1)
        value,prediction=torch.max(poss,dim=1)
        percent_poss=value.item()*100
        plt.figure(facecolor='#1E1E1E')
        plt.imshow(img)
        plt.title(f"This is a/an {label_dict[prediction.item()]} with a {percent_poss:.2f}% possibility", color='white')
        plt.axis('off')
        plt.savefig(r"assets/model-output.png",facecolor='#1E1E1E')
        plt.show()
        
if __name__=="__main__":
    img_tensor=getimg(img)
    pred(img_tensor,img)
        
    
    
    
    
    
    

