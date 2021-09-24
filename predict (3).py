# Import Modules
import argparse
import torch
import os
import time
import torch
import PIL
import json
import numpy as np

from math import ceil
from PIL import Image
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description = 'Process a series of integers')
    parser.add_argument('--image_path', type = str, help = 'image path that is to be predicted')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'path of trained model')
    parser.add_argument('--topk', default = 5, help = 'display the top 5 k probabilities')
    parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name.json', help = 'pathway to flower names')
    args = parser.parse_args()
    return args

# Load a previously saved checkpoint
def load_checkpoint(filepath):
    if os.path.isfile(filepath):
        print("Currently loading checkpoint '{}'".format(filepath))
        
        checkpoint = torch.load(filepath)
        #model = getattr(models, checkpoint['arch'])(pretrained = True)
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
        print("checkpoint load complete")
        return model
        
def process_image(image):
    
    # From PIL import image path
    im = Image.open(image)
    
    # dimensions
    w = 256
    h = 256
    im = im.resize((w, h))
    
    new_width = 224
    new_height = 224
    
    left = (w - new_width)/2
    top = (h - new_height)/2
    right = (w + new_width)/2
    bottom = (h + new_height)/2
    
    # cropping
    im = im.crop((left, top, right, bottom))
    
    # Normalize
    np_image = np.array(im)/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # subtract means and divide by standard deviation
    np_image = (np_image - mean)/std
    
    # transpose image
    image = np_image.transpose((2, 0, 1))
    
    return image

def predict(image_path, model, topk=5):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    
    image = torch.tensor(process_image(image_path)).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    #load_checkpoint('checkpoint.pth')
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    ps_topk_class = []
    
    with torch.no_grad():
        #image = process_image(image_path).unsqueeze(dim=0)
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        
        # takes top 5 probabilities and index from the output
        ps_topk = ps.cpu().topk(topk)[0].numpy()[0]
        ps_topk_index = ps.cpu().topk(topk)[1].numpy()[0]
        
        # Loop through class_to_idx dictionary to reverse key, values in idx_to_class dictionary
        for key, value in model.class_to_idx.items():
            model.idx_to_class[value] = key
        
        # Loop through index to retrieve class from idx_to_class dict
        for item in ps_topk_index:
            ps_topk_class.append(model.idx_to_class[item])
    
    print(ps_topk)
    print(ps_topk_class)
    
    return ps_topk, ps_topk_class

def main():
    global args
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_checkpoint(args.checkpoint)
    
    #image_tensor = process_image(args.image_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    image_path = args.image_path
    
    prediction = predict(image_path, model, topk=5)
    
    print("Prediction has been completed!")
    return prediction

if __name__ == '__main__':
    main()


    
                        
    
    