import numpy as np
import pandas as pd
import librosa
import os
import re
from sklearn.metrics import roc_curve
import csv 
import datetime
from sklearn.preprocessing import StandardScaler


def calculate_eer(labels, scores):
    '''
        calculating EER of Top-S detector
        
        Args:
            labels(Boolean or int): vector 1 is postive, 0 is negative.
            scores: Vector(float) 
        
        Returns:
            EER(float): Equal Error Rate.
    '''

    # Calculating EER
    fpr,tpr,threshold = roc_curve(labels, scores, pos_label=0)
    fnr = 1-tpr
    # EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    return EER 


def record_eer(path, batchsize, lr, epoch, eer, model='', feature=''):
    '''

        Args:
            path(str): Path to log.
            batchsize(int): batchsize of the model
            lr(float): learning rate of the model
            epoch(int): epoch of the training
            eer(float): EER(%) result
            model(str): Model name
            feature(str):feature type

    '''
    time = datetime.datetime.now()
    time = time.strftime('%Y/%d/%m %H:%M')
    data = [path, model, eer, batchsize, lr, epoch,  time, feature]
    with open('/home/ozora/audio/asvspoof/2ch/logs/eer_logs.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(data)


def save_score(path, score):
    with open(path, 'w') as f:
        for v in score:
            v = str(v)
            f.write(v+'\n')




def txt_to_float(path):
    with open(path, 'r') as f:
        txt = f.read()

    data = txt.split()
    scores = np.zeros(len(data))

    for i, value in enumerate(data):
        scores[i] = float(value)
    
    return scores
    
