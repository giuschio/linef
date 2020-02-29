# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:53:32 2020

@author: Giulio Schiavi
"""

from util import generateData as gd
from util import fitLine as fit
from net import Net
import torch
import torch.optim as optim
from random import randint
from util import mprint as mprint
from util import almost_equals as almost_equals

datadim = 29

def rd(n):
    return list(set([randint(0,datadim) for i in range(n)]))

number_training = 1000
number_batch = 50
number_hypotheses = 10
number_epochs = 1

net = Net()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
training_set = gd(number_training)

def length(lt):
    if(len(lt) % 2 == 0):
        return len(lt)
    return len(lt) - 1

avg_loss = torch.zeros(1,1)
train = 1
if(train):
    for e in range(number_epochs):
        t = 0
        while(1):
            if(t > number_training-2):
                break
            optimizer.zero_grad()
            LOSS = torch.zeros(1,1)
            for b in range(number_batch):
                example = training_set[t]
                x = example.x[0,:]
                y = example.y[0,:]
                gm = example.m
                gq = example.q
                
                #pick 14 indexes at random from 0 to 29
                #cast to set to avoid repetition
                indeces = rd(14)
                L = length(indeces)
                scores = torch.zeros(1,L//2)
                loss = torch.zeros(1,L//2)
                for i in range(0,L,2):
                    x1,y1,x2,y2 = x[indeces[i]], y[indeces[i]], x[indeces[i+1]], y[indeces[i+1]]
                    #fit a line to the two points
                    m,q = fit(x1,x2,y1,y2)
                    #now count the inliers
                    inliers = torch.FloatTensor([m*x[a] + q - y[a] == 0 for a in range(30)])
                    #now the score, which is going to be some function of the inliers
                    scores[0][i//2] = net(inliers)
                    #now calculate the loss
                    loss[0][i//2] = (m - gm)**2 + (q - gq)**2
                #finally calculate the expected loss of this example
                scores_n = torch.softmax(scores[0,:], dim = 0)
                exp_loss = loss[0,:].dot(scores_n)
                #print(exp_loss)
                LOSS += exp_loss
                t += 1
                #print(t)
            LOSS.backward()
            optimizer.step()
            avg_loss += LOSS
            if(t % 100 == 0):
                print(avg_loss/100)
                avg_loss = torch.zeros(1,1)

        
print('net trained')
print('now testing')


number_test = 30
test_set = gd(number_test)
number_correct = 0
ε = 1e-4
with(torch.no_grad()):
    for t in range(number_test):
        example = test_set[t]
        x = example.x[0,:]
        y = example.y[0,:]
        gm = example.m
        gq = example.q
        
        #pick 14 indexes at random from 0 to 29
        #cast to set to avoid repetition
        indeces = rd(14)
        L = length(indeces)
        maxscore = -10000
        maxparams = 0
        for i in range(0,L,2):
            x1,y1,x2,y2 = x[indeces[i]], y[indeces[i]], x[indeces[i+1]], y[indeces[i+1]]
            #fit a line to the two points
            m,q = fit(x1,x2,y1,y2)
            #now count the inliers
            inliers = torch.FloatTensor([m*x[a] + q - y[a] == 0 for a in range(30)])
            #now the score, which is going to be some function of the inliers
            score = net(inliers)
            if(score > maxscore):
                maxscore = score
                mm = m.item()
                mq = q.item()
        #mprint([mm,mq], [gm,gq], maxscore)
        if(almost_equals(mm,mq,gm,gq,ε)):
            number_correct += 1
            
print('% correct: ' + str(number_correct/number_test * 100))

            
    