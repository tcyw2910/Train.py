# Import Modules
import argparse
import torch
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

def parse():
    parser = argparse.ArgumentParser(description = 'Train neural network')
    parser.add_argument('--data_directory', default="flowers", help = 'data directory is required!')
    parser.add_argument('--save_dir', help = 'directory needed to save a neural network')
    parser.add_argument('--arch', help = 'model to use')
    parser.add_argument('--learning_rate', type = float, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, help = 'No. of hidden units')
    parser.add_argument('--epochs', type = int, help = 'Epochs')
    parser.add_argument('--gpu', action = 'store_true', help = 'gpu')
    args = parser.parse_args()
    return args

# define transforms for training, validation, and testing sets
def def_data(data_dir):
    print("Data processed into training, test and validation data, as well as labels")
    train_dir, test_dir, valid_dir = data_dir
    train_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
                       
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print("Training dataset images:", len(train_data))
    print("Training dataset batches:", len(train_loader))
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':train_loader, 'valid':validation_loader, 'test':test_loader, 'labels':cat_to_name}
    return loaders

# Data
def obtain_data():
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'
    data_dir = [train_dir, test_dir, valid_dir]
    return def_data(data_dir)

# Pre-trained model
def specify_model(data):
    if (args.arch is None):
        #arch_type = 'densenet'
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    
    #if (arch_type =='densenet'):
        #model = models.densenet121(pretrained = True)
        #model.name = "densenet121"
        #input_node = 1024
       #output_node = 102
        
    if (arch_type == 'vgg'):
        model = models.vgg16(pretrained = True)
        model.name = "vgg16"
        input_node = 25088
        output_node = 500
    if (args.hidden_units is None):
        hidden_units = 512
    else:
        hidden_units = arg.hidden_units
    
    # Freeze parameters so we don't backprop
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_units = int(hidden_units)
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),  
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    # Use GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    model.to(device);
    return model

# Do Validation on test set
def validation(model, testloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    accuracy = 0
    test_loss = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        
        # Calculate Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, int(accuracy)

def train(model, data):
    print("Training Model")
    
    print_every = 50
    steps = 0
    
    if (args.learning_rate is None):
        learn_rate = 0.003
    else:
        learn_rate = args.learning_rate
    if(args.epochs is None):
        epochs = 2
    else:
        epochs = args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = data['train']
    validation_loader = data['valid']
    test_loader = data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    
    model.to(device);
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            steps += 1
        
            # move images and label tensors to the default device 
            images, labels = images.to(device), labels.to(device)
        
            # training pass
            optimizer.zero_grad()
        
            # make a forward pass
            output = model.forward(images)
        
            # calculate loss using the logits
            loss = criterion(output, labels)
        
            # perform a backward pass
            loss.backward()
        
            # take a step w/ optimizer to update the weights
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.."
                  f"Train Loss: {running_loss/print_every:.3f}.."
                  f"Test Loss: {test_loss/len(validation_loader):.3f}.."
                  f"Test Accuracy: {accuracy/len(validation_loader):.3f}")
            
                running_loss = 0
                model.train() 
                
    print("\n Training has been completed!")
    return model

def save_checkpoint(model, train_data):
    print("Saving the current model")
    if (args.save_dir is None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    
    # Assign class_to_idx as attribute to model
    model.class_to_idx = train_data.class_to_idx  
    
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'epochs': 2,
                  'hidden_layers': 4096,
                  'learning_rate': 0.003,
                  'arch': "vgg16",
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()
                 }
    
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = obtain_data()
    
    model = specify_model(data)
    model = train(model, data)
    train_dir = args.data_directory + '/train'
    train_data = train_transform(train_dir)
    save_checkpoint(model, train_data)
    
def main():
    
    model = models.vgg16(pretrained = True)
    model.name = "vgg16"
    print("Image classifier is being created!")
    global args
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = args.data_directory + '/train'
    train_data = train_transform(train_dir)
    model = specify_model(train_data)
    model.to(device)
    create_model()
    print(model)
    return None

if __name__ == "__main__":
    main()
        
       
   


    


