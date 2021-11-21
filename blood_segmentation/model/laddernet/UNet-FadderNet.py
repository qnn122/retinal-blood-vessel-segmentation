#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[1]:


import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.metrics.ranking import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fadder_net import *


# In[2]:


import torchvision
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# # Building network

# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LadderNetv6(inplanes=1).to(device)


# In[ ]:


net


# # Dataset loader class

# In[4]:


class TSegLoader(torch.utils.data.Dataset):
    def __init__(self, image_names, image_folder, mask_folder):
        self.images = image_names
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        
        self.mx = torchvision.transforms.Compose([
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x : torch.cat([x,1-x], dim=0))
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        i1 = Image.open(os.path.join(self.image_folder,self.images[i]))
        m1 = Image.open(os.path.join(self.mask_folder,self.images[i])).convert("1")
        
        return self.tx(i1), self.mx(m1)


# # Build data loader

# In[5]:


image_folder = 'patches48/images'
mask_folder = 'patches48/masks'
image_name = [filename for filename in os.listdir(image_folder) if not filename.startswith('.') and filename.endswith('.jpg')]


# In[6]:


train_images, validation_images = train_test_split(image_name, test_size=0.1, random_state = 2019)


# In[7]:


batch_size=128
train_loader = torch.utils.data.DataLoader(TSegLoader(train_images,image_folder,mask_folder), batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(TSegLoader(validation_images,image_folder,mask_folder), batch_size=batch_size, shuffle=True)


# In[8]:


opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
loss_function = nn.MSELoss()
scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 5, mode = 'min', verbose=True)
best_loss = 999999999


# # Load best model

# In[9]:


model_path = 'models/faddernet-patch48.pth'
if os.path.isfile(model_path):
    model_checkpoint = torch.load(model_path)
    net.load_state_dict(model_checkpoint['state_dict'])
    opt.load_state_dict(model_checkpoint['optimizer'])
    best_loss = model_checkpoint['best_loss']
    print(f"Loaded previous checkpoint with loss {best_loss}\n\n")
#else:
    #get_ipython().system(u'mkdir models')


# # Training phase

# In[ ]:


N_EPOCHES = 250
train_losses, val_losses = [], []

for e in range(N_EPOCHES):
    running_loss = 0
    net.train()
    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        opt.zero_grad()
        y_pred = net(images)            
        loss = loss_function(y_pred, masks)
        loss.backward()            
        opt.step()
        running_loss += loss.item()
        try:
            train_roc_score = roc_auc_score(masks.to('cpu').numpy().reshape(-1).astype(np.int), y_pred.to('cpu').detach().numpy().reshape(-1))
        except:
            train_roc_score = -1
        sys.stdout.write(f"\rEpoch {e+1}/{N_EPOCHES}... Training step {step+1}/{len(train_loader)}... Loss {running_loss/(step+1)}... AUC {train_roc_score}")
    else:
        print()
        val_loss = 0
        val_roc_score = 0
        net.eval()
        with torch.no_grad():                
            for step, (images, masks) in enumerate(validation_loader):                    
                images, masks = images.to(device), masks.to(device)                  
                y_pred = net(images)
                val_loss += loss_function(y_pred, masks)
                try:
                    val_roc_score += roc_auc_score(masks.to('cpu').numpy().reshape(-1).astype(np.int), y_pred.to('cpu').numpy().reshape(-1))
                except:
                    val_roc_score += 1
                sys.stdout.write(f"\rEpoch {e+1}/{N_EPOCHES}... Validating step {step+1}/{len(validation_loader)}... Loss {val_loss/(step+1)}... AUC ROC {val_roc_score/(step+1)}")
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(validation_loader))
        print()
        scheduler.step(val_loss/len(validation_loader))
        print("Epoch: {}/{}.. ".format(e+1, N_EPOCHES),                  
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),                  
              "Val Loss: {:.3f}.. ".format(val_loss/len(validation_loader)),
              "AUC ROC: {:.3f}.. ".format(val_roc_score/len(validation_loader)))
        if best_loss > val_loss/len(validation_loader):
            print("Improve loss of model from {} to {}".format(best_loss, val_loss/len(validation_loader)))
            best_loss = val_loss/len(validation_loader)
            torch.save({'epoch': e + 1, 'state_dict': net.state_dict(), 'best_loss': best_loss, 'optimizer' : opt.state_dict()}, model_path)
        print()




