# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:54:30 2020

@author: Giulio Schiavi
"""
from random import random
import torch

def rd(lo, hi):
    #normalized random that also produces negative values
    size = hi-lo
    return random()*size + lo

class DataSet():
    def __init__(self, dimension, percentage_outliers):
        self.m = rd(-2,2)
        self.q = rd(-2,2)
        self.x = torch.rand(1,dimension)*100 - 50
        self.y = self.x * self.m + self.q
        for i in range(dimension):
            if(rd(0,1) < percentage_outliers):
                self.y[0][i] = rd(-50,50)
                
    def ground(s):
        return [s.m, s.q]


def generateData(nsets, npoints = 30, percentage_outliers = .10):
    #number of data sets, number of (x,y) points per data set, percentage of outliers in each dataset
    return [DataSet(npoints, percentage_outliers) for i in range(nsets)]
