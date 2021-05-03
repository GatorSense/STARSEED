# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:26:08 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from barbar import Bar

    
def train_model(model, dataloaders, criterion, optimizer, device, 
                          saved_bins=None, saved_widths=None,histogram=True,
                          num_epochs=25,scheduler=None,dim_reduced=False):
    since = time.time()

    test_acc_history = []
    train_acc_history = []
    train_error_history = []
    test_error_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and testidation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                if(histogram):
                    if dim_reduced:
                        #save bins and widths
                        saved_bins[epoch+1,:] = model.histogram_layer[-1].centers.detach().cpu().numpy()
                        saved_widths[epoch+1,:] = model.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                    else:
                        #save bins and widths
                        saved_bins[epoch+1,:] = model.histogram_layer.centers.detach().cpu().numpy()
                        saved_widths[epoch+1,:] = model.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'test':
                test_error_history.append(epoch_loss)
                test_acc_history.append(epoch_acc)

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))               
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #Returning error (unhashable), need to fix
    train_dict = {'best_model_wts': best_model_wts, 'test_acc_track': test_acc_history, 
                  'test_error_track': test_error_history,'train_acc_track': train_acc_history, 
                  'train_error_track': train_error_history,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
          
def test_model(dataloader,model,device):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels,index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
        
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / (len(dataloader.sampler))
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2)}
    
    return test_dict

