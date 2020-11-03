#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:31:02 2020

@author: matthewconlin
"""

# Standard library imports #
import math

# Third-party imports #
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier



class PTZViewSepML():
    
    '''
    Create and execute a machine learning object (train and apply) for automatically separating views from a PTZ camera.
    
    Inputs:
        vidList (list): a list of video files from which to extract frames 
        percentTraining (int): Percent of input frames to use for training (rest will be used or applying)
        viewWant (int): The number of the desired view
        existingDataAr (array): Existing aray containing feature vectors and view codes, derived from using this class before.
    '''
    
    def __init__(self,vidList,percentTraining,viewWant,existingDataAr=None):
        self.vidList= vidList
        self.percentTraining = percentTraining
        self.viewWant = viewWant
        if existingDataAr.any():
            self.existingDataAr = existingDataAr
        else:
            self.existingDataAr = [0]
                
        
    def train(self):
        
        '''
        Train the random forest classifier by manually identifying the view code on a bunch of frames 
        '''
        
        if len(self.existingDataAr) == 1:
            #=========================#
            # Manually identify view category for a bunch of frames and save their feature vectors as a big array  
            #=========================#  
            dataList = []
            allDataList = []
            for vid in self.vidList:
                          
                self.GetVidData(vid)
                dataList.append([self.fvec,self.category])
                
                self.makeDataAr(dataList)
                allDataList.append(self.dataAr)
            self.allDataAr = np.concatenate(allDataList,axis=0)
        else:
            self.allDataAr = self.existingDataAr
               
        #=========================#
        # Use the data to create and train the Random Forest classifier # 
        #=========================#  
        self.CreateAndEvalRandomForest(self.allDataAr,self.percentTraining,self.viewWant)
        
        return self.clf,self.dataAr_filled,self.accuracy,self.numObs,self.numMissed,self.numFalse
    
    
    
    def apply(self,vid):
        
        self.ApplyRandomForest(self.clf,vid,self.viewWant)
#        self.CheckViewSep(vid,self.framesKeep)
#
#        framesUse = [i for i in self.framesKeep if i not in self.toRemove]
        framesUse = self.framesKeep
        
        return framesUse
    
    

    def GetVidData(self,vid):
        
        '''
        Display a series of frames from an input video, allowing the user to specify the
        numeric code of that image's view and saving the color feature vector of the image.
        Note: If using Spyder this works best if the figure display mode is set to 'Inline'.
        '''
    
        # Read frames of the videos, classify them, and save the feature vectors #
        cap = cv2.VideoCapture(vid)
        numFrames = int(cap.get(7))
        fvec = list()
        category = list()
        for i in range(0,numFrames,50): # Take every 50th frame (2% of frames)
            
            for attempt in range(0,5): # Give the user 5 tries to not make a mistake, and allow a retry if they do. #
                try:
                    cap.set(1,i) 
                    test,im = cap.read()
                     
                    # Find the longest line in the frame #
                    kernel = np.ones((5,5),np.uint8)
                    imeroded = cv2.erode(im,kernel,iterations=1)
                    edges = cv2.Canny(imeroded,50,100)
                    lines = cv2.HoughLines(edges,1,np.pi/180,200)
                    if lines is not None:
                        for rho,theta in lines[0]: # The first element is the longest edge #
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a*rho
                            y0 = b*rho
                            x1 = int(x0 + 1000*(-b))
                            y1 = int(y0 + 1000*(a))
                            x2 = int(x0 - 1000*(-b))
                            y2 = int(y0 - 1000*(a))
                        psi = math.atan2(y1-y2,x2-x1)
                    else:
                        psi = 0
                    
                    
                    ff = [np.mean(im[:,:,2]),np.mean(im[:,:,1]),np.mean(im[:,:,0]),np.std(im[:,:,2]),np.std(im[:,:,1]),np.std(im[:,:,0]),psi]
                    fvec.append(ff) # Feature vector of means and stdevs of each color #
                    
                    self.GetViewNumber(np.flip(im,axis=2))
                    if self.cat != 'Quit':
                        category.append(int(self.cat)) # Establish the view category of the image #
                    else:
                        return
                        
                except:
                     print('An error occured. Please re-input the view code: ')
                else:
                     break
            else:
                break
                 
        self.fvec = fvec
        self.category = category

    
    def GetViewNumber(self,im):
        plt.figure()
        plt.imshow(im)
        plt.show()
        cat = input('Which view is this?: ')
        self.cat = cat   
    
    
    def makeDataAr(self,dataList):
        dataAr = np.empty([0,1+len(dataList[0][0][0])])
        for ii in range(0,len(dataList)):
            for i in range(0,len(dataList[0][1])):
                row = np.hstack([dataList[ii][0][i],dataList[ii][1][i]])
                dataAr = np.vstack([dataAr,row])
        
        self.dataAr = dataAr
            
    
    
    def CreateAndEvalRandomForest(self,dataAr,percentTraining,viewWant): 
        
        '''
        Uses input data to create a random forest classifier, and evaluates it based on observations
        witheld from the training.
        
        Inputs:
            dataAr: a numpy array containing (as columns) the view classifications and associated
            feature vectors (e.g. as returned from GetVidData). The array also needs to contain a 
            column of nans. Predicted view categories will replace nans in this column for entries
            used for model evaluation. The array can have as many other columns as you want 
            (e.g. metadata to assist in anaysis such as dates), you just need to pass the
            numbers of the columns that contain the view category and feature vector elements. 
        '''
    
        # Establish the observations to use as training data. Take linearly spaced entries of x% of the total entries #       
        iTrain = np.round(np.linspace(0,len(dataAr)-1,round(len(dataAr)*(percentTraining/100))))
        iTrain = np.array([int(i) for i in iTrain])
        
        dataTrain = dataAr[iTrain,:]
#        dataTrain = np.empty([0,len(dataAr)])
#        for i in iTrain:
#            dataTrain = np.vstack([dataTrain,np.hstack([dataAr[i,catColumn],dataAr[i,fvecColumns]])])
                
        # Train the algorithm #
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(dataTrain[:,0:len(dataTrain[0,:])-1],dataTrain[:,len(dataTrain[0,:])-1])
        
        # Predict the view of all the frames not used for training and put into the dataAr #
        iTest = [i for i in np.arange(0,len(dataAr)-1) if i not in iTrain]
        predv = np.zeros(len(dataAr))*np.nan
        for i in iTest:
            fv = dataAr[i,0:len(dataTrain[0,:])-1]
            catPred = self.clf.predict([fv])
            predv[i] = catPred
        self.dataAr_filled = np.hstack([dataAr,np.reshape(predv,[len(predv),1])])
            
    
        # Evaluate the accuracy of the tree #
        obs = self.dataAr_filled[~np.isnan(self.dataAr_filled[:,len(self.dataAr_filled[0,:])-1]),len(self.dataAr_filled[0,:])-2]
        pred = self.dataAr_filled[~np.isnan(self.dataAr_filled[:,len(self.dataAr_filled[0,:])-1]),len(self.dataAr_filled[0,:])-1]
        
        self.accuracy = 100-(len(np.where(obs!=pred)[0])/len(obs)*100)
    
        iView_obs = np.where(obs == viewWant)[0]
        iView_pred = np.where(pred == viewWant)[0]
        self.numObs = len(iView_obs)
        self.numMissed = len([i for i in iView_obs if i not in iView_pred])
        self.numFalse = len([i for i in iView_pred if i not in iView_obs])
    
      
        
    def ApplyRandomForest(self,clf,vid,viewWant):
    
        # Get video info #
        cap = cv2.VideoCapture(vid)
        numFrames = int(cap.get(7))
        
        # Read each frame, create its feature vector, and classify it with the random forest. #
        framesKeep = list()
        for i in range(0,numFrames):        
            cap.set(1,i) 
            test,im = cap.read()
            
            # Find the longest line in the frame #
            kernel = np.ones((5,5),np.uint8)
            imeroded = cv2.erode(im,kernel,iterations=1)
            edges = cv2.Canny(imeroded,50,100)
            lines = cv2.HoughLines(edges,1,np.pi/180,200)
            if lines is not None:
                for rho,theta in lines[0]: # The first element is the longest edge #
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                psi = math.atan2(y1-y2,x2-x1)
            else:
                psi = 0
            
            fv = [np.mean(im[:,:,2]),np.mean(im[:,:,1]),np.mean(im[:,:,0]),np.std(im[:,:,2]),np.std(im[:,:,1]),np.std(im[:,:,0]),psi]
            
            catPred = clf.predict([fv])
            
            # Keep the frame num if this frame was classified as the view we want #
            if int(catPred) == int(viewWant):
                framesKeep.append(i)
                
        self.framesKeep = framesKeep
    


    def CheckViewSep(self,vid,framesKeep):
        
        
        def checkFrames(cap,frameList):
            s = plt.subplots(5,3)[1]
           
            numm = -1
            for sub in s:
               for subsub in sub:
                   numm = numm+1
                   try:
                       cap.set(1,frameList[numm])
                       test,im = cap.read()
                       
                       subsub.imshow(np.flip(im,axis=2))
                       subsub.text(len(im[:,:,0])/2,len(im[:,:,1])/2,str(frameList[numm]))
                   except IndexError:
                       plt.show()
                       iBad = input('Which frame numbers ARE NOT part of the view you want to keep (separate multiple with commas)? Press Enter if none: ')
                       return iBad
                   
            plt.show()
            iBad = input('Which frame numbers ARE NOT part of the view you want to keep (separate multiple with commas)? Press Enter if none: ')
            
            return iBad
    
         
        # Get video info #
        cap = cv2.VideoCapture(vid)
        
        toRemove = list()
        num = 0
        while num<len(framesKeep):
            
            frameList = framesKeep[num:num+15]
            
            iBad = checkFrames(cap,frameList)
    
            if iBad:
                if ',' in iBad:
                    iBad_int = [int(i) for i in iBad.split(',')]
                    [toRemove.append(i) for i in iBad_int]
                else:
                    iBad_int = int(iBad)
                    toRemove.append(iBad_int)
            else:
                pass
            
            num = num+15
            
        self.toRemove = toRemove
        
 

     

def CreatePTZImageProduct(vid,framesUse,product):
    
    ''' 
    Create an image product from a PTZ camera.
    
    Inputs:
        vid (str): The video file from which to ceate the image product.
        framesUse (list): The frames from the video that are of the wanted view. Found from using the PTZViewSepML class.
        product (int): The code for the image product to create. Currently only a code of 1, for timex, is supported. 
        
    Returns:
        product (array): The image product data
    
    '''
    
    def addFrame(r,g,b,r_list,g_list,b_list):
        r = r+r_list[0]
        g = g+g_list[0]
        b = b+b_list[0]
        
        r_list = r_list[1:len(r_list)]
        g_list = g_list[1:len(g_list)]
        b_list = b_list[1:len(b_list)]
        
        return r,g,b,r_list,g_list,b_list
            
    
    # Read a frame to get the size #
    cap = cv2.VideoCapture(vid)
    cap.set(1,1) 
    test,im = cap.read()
    
    # Initialize empty arrays for each color channel #
    r = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    g = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    b = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    
    # Loop through each frame to keep and add its color channel values to the growing array #
    r_list = list()
    g_list = list()
    b_list = list()
    for frame in range(0,len(framesUse)):

        cap.set(1,framesUse[frame]) 
        test,image = cap.read()
        
        r_list.append(image[:,:,2])
        g_list.append(image[:,:,1])
        b_list.append(image[:,:,0])
        
    # Create the desired product #
    if product == 1: # Timex #
        
        while len(r_list)>0:
            r,g,b,r_list,g_list,b_list = addFrame(r,g,b,r_list,g_list,b_list)
        
        del r_list,g_list,b_list
        
        r = r/len(framesUse)
        g = g/len(framesUse)
        b = b/len(framesUse)
            
        product = np.stack([r/255,g/255,b/255],axis=2)
        
        return product

    elif product == 2: # Variance #
        r = np.stdev(r)



def CreateImageProduct(vid,product):

    ''' 
    Create an image product from a (web)camera.
    
    Inputs:
        vid (str): The video file from which to ceate the image product.
        product (int): The code for the image product to create. Currently only a code of 1, for timex, is supported. 
        
    Returns:
        product (array): The image product data
    
    '''    
    def addFrame(r,g,b,r_list,g_list,b_list):
        r = r+r_list[0]
        g = g+g_list[0]
        b = b+b_list[0]
        
        r_list = r_list[1:len(r_list)]
        g_list = g_list[1:len(g_list)]
        b_list = b_list[1:len(b_list)]
        
        return r,g,b,r_list,g_list,b_list
            
    
    # Read a frame to get the size #
    cap = cv2.VideoCapture(vid)
    cap.set(1,1) 
    test,im = cap.read()
    
    # Initialize empty arrays for each color channel #
    r = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    g = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    b = np.empty((len(im[:,0,:]),len(im[0,:,:])))
    
    # Loop through each frame to keep and add its color channel values to the growing array #
    r_list = list()
    g_list = list()
    b_list = list()
    numFrames = int(cap.get(7))
    for frame in range(0,numFrames):

        cap.set(1,frame) 
        test,image = cap.read()
        
        r_list.append(image[:,:,2])
        g_list.append(image[:,:,1])
        b_list.append(image[:,:,0])
        
    # Create the desired product #
    if product == 1: # Timex #
        
        while len(r_list)>0:
            r,g,b,r_list,g_list,b_list = addFrame(r,g,b,r_list,g_list,b_list)
        
        del r_list,g_list,b_list
        
        r = r/numFrames
        g = g/numFrames
        b = b/numFrames
            
        product = np.stack([r/255,g/255,b/255],axis=2)
        
        return product

    elif product == 2: # Variance #
        r = np.stdev(r)

