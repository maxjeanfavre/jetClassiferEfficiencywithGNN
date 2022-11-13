import torch
from sklearn.metrics import confusion_matrix
import numpy as np

class my_metrics():
    @staticmethod
    def accuracy(model_out, target):
        #print("model_out ",model_out[0])
        #print("target ",target[0])
        pred = torch.argmax(model_out, dim=1)
        #print("pred ",pred[0])
        cm = confusion_matrix(target.cpu(), pred.cpu()) 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm = confusion_matrix(target.cpu(), pred.cpu()) 
        #return cm.diagonal()/cm.sum(axis=1)   
        return cm.diagonal() 
