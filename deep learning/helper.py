import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

def load_data():
    data_dir = './flowers/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]) 
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456,0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir,transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir,transform = test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
    return train_data,trainloader,testloader,validloader

def load_json(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def build_model(gpu,arch,hidden_units,learning_rate):
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(25088,hidden_units),
                          nn.ReLU(),
                          nn.Dropout(p=0.1),
                          nn.Linear(hidden_units,1024),
                          nn.ReLU(),
                          nn.Dropout(p=0.1),
                          nn.Linear(1024,102),
                          nn.LogSoftmax(dim=1))
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(1024,hidden_units),
                          nn.ReLU(),
                          nn.Dropout(p=0.1),
                          nn.Linear(hidden_units,256),
                          nn.ReLU(),
                          nn.Dropout(p=0.1),
                          nn.Linear(256,102),
                          nn.LogSoftmax(dim=1))
    else:
        print('Sorry, the architecture is not available.')
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)
    model.to(device)
    return model,device,criterion,optimizer

def train_model(epochs,trainloader,validloader,model,device,criterion,optimizer):
    epochs = epochs
    steps = 0
    running_loss =0
    print_every = 50
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        
            inputs, labels = inputs.to(device),labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps% print_every ==0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps,labels)
                    
                        validation_loss +=batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p,top_class = ps.topk(1,dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
def save_checkpoint(model,epochs,arch,optimizer,train_data):
    checkpoint = {'epochs':epochs,
                  'architecture':arch,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict,
                  'state_dict': model.state_dict(),
                  'class_to_idx':train_data.class_to_idx           
             }

    torch.save(checkpoint,'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif checkpoint['architecture'] =='densenet121':
        model = models.densenet121(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))])
    pil_image = image_transform(im)
    return pil_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, cat_to_name,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda:0')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image =image.float()
    
    with torch.no_grad():
        logps = model.forward(image.cuda())
        ps = torch.exp(logps)
        top_p,top_index = ps.topk(topk,dim = 1)
        top_class = []
        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        for key in top_index.tolist()[0]:
            top_class.append(idx_to_class[key])
        labels = []
        for key in top_class:
            labels.append(cat_to_name[key])    
    return top_p.tolist()[0],labels