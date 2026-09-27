from functions import *
from classes import *
#WHOLE TRAINING PROCESS STEP BY STEP
def train():
    # 1 - Define the device
    device=("cuda"if torch.cuda.is_available()else "cpu")

    # 2 - Define the model
    model=CNN().to(device)
    
    # 3 - Get train and test loaders
    train_loader,test_loader=get_data_loaders()
    
    # 4 - Get criterion and optimizer
    criterion,optimizer=define_loss_and_optimizer(model)
    
    # 5 - Deploy Scheduler (in order to stop overfitting by decreasing learning rate progressively after each epoch)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
    
    # 6 - Start training
    train_model(model,5,train_loader,criterion,optimizer,scheduler,epochs=30)
    
    # 7 - Test the model with both test and train datasets. 
    train_accuracy=float(test_model(model,train_loader))
    test_accuracy=float(test_model(model,test_loader))
    
    # 8 - Check both values in order to detect possible overfitting
    print(f"Train Accuracy: {train_accuracy} % \n Test Accuracy: {test_accuracy} % ")
    if train_accuracy-test_accuracy>10:
        print(f"Possible Overfitting has been detected. You might consider changing epoch number.")

if __name__=="__main__":
    train()
