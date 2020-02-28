# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:54:30 2020

@author: Giulio Schiavi
"""
from random import random

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
class DataSet():
    def __init__(self, dimension):
        self.m = 0
        self.q = 0
        self.data = [0]*dimension
        self.dim = dimension
        self.last = 0
        self.it = 0
    
    def append(self, point):
        self.data[self.last] = point
        self.last += 1
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.it < self.dim:
            self.it += 1
            return self.data[self.it-1]
        raise StopIteration
        
        
        
        
def rd(lo, hi):
    #normalized random that also produces negative values
    size = hi-lo
    return random()*size + lo
        

def _generateData_u(npoints, noutliers = 10):
    data = DataSet(npoints)
    data.m = rd(-2,2)
    data.q = rd(-5,5)
    ninliers = npoints - noutliers
    #generate data points in a 100x100 window centered in 0
    #i.e. points will have coordinates x ∈ (-50,50), same with y
    for i in range(ninliers):
        x = rd(-50,50)
        y = data.m * x + data.q
        data.append(Point(x,y))
    for i in range(noutliers):
        x = rd(-50,50)
        y = rd(-50,50)
        data.append(Point(x,y))
    return data
    

def generateData(nsets, npoints = 30, noutliers = 3):
    #number of data sets, number of (x,y) points per data set, percentage of outliers in each dataset
    bag = [0]*nsets
    for i in range(nsets):
        bag[i] = _generateData_u(npoints, noutliers)
    return bag