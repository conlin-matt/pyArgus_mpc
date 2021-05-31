#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:49:52 2020

@author: matthewconlin
"""

# Standard-library imports #
import math

# Third-party imports #
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi



def calcxy(calibVals,X,Y,Z):
    
    '''
    Calculate the image (xy) coordinates from given real-world (XYZ) coordinates based on geometry solution/calibration parameters.
    
    Inputs:
        calibVals (array or list): The calibration parameters (aka the geometry solution, but also including some intrinsics) in the order: omega, phi, kappa, camX, camY, camZ, f, x0, y0.
        X (float): Real-world X coordinate of point
        Y (float): Real-world Y coordinate of point
        Z (float): Real-world Z coordinate of point 
        
    Returns:
        x,y: The image coordinates of the point based on your geometry solution and calibration values. 
        
    Created by:
        Matthew P. Conlin, University of Florida
        05/2020
    '''
    
    
    # Define the calib params #
    omega = calibVals[0]
    phi = calibVals[1]
    kappa = calibVals[2]
    XL = calibVals[3]
    YL = calibVals[4]
    ZL = calibVals[5]
    f = calibVals[6]
    x0 = calibVals[7]
    y0 = calibVals[8]
    
    # Set up the rotation matrix #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
        
    # Get image coordinates of each desired world coordinate based on calib vals #
    x = x0 - (f*(((m11*(X-XL)) + (m12*(Y-YL)) + (m13*(Z-ZL))) / ((m31*(X-XL)) + (m32*(Y-YL)) + (m33*(Z-ZL)))))
    y = y0 - (f*(((m21*(X-XL)) + (m22*(Y-YL)) + (m23*(Z-ZL))) / ((m31*(X-XL)) + (m32*(Y-YL)) + (m33*(Z-ZL)))))
    
    return x,y



def calcXYZ(calibVals,x,y,Z):
    
    '''
    Calculate the real-world (XY) coordinates of a point given the image (xy) coordinates and the geometry solution/calibration parameters.
    To do so, the collinearity conditions are solved analytically. This is not excessively complicated but is messy. A derivation for the solution 
    using the terminology in this function can be found at: https://drive.google.com/file/d/1HrWPLsQnw-4p2_Hef9ilJNGUo2qJ0PIX/view?usp=sharing
    
    Inputs:
        calibVals (array or list): The calibration parameters (aka the geometry solution, but also including some intrinsics) in the order: omega, phi, kappa, camX, camY, camZ, f, x0, y0.
        x,y (float): Image coordinates of point
        Z (float): Real-world Z coordinate of point 
        
    Returns:
        X,Y: The real-world coordinates of the point based on your geometry solution and calibration values. 
        
    Created by:
        Matthew P. Conlin, University of Florida
        05/2020
    '''
    
    #=========================================================================================#    
    #1. Break out the calibration parameters and calculate the elements of the rotation matrix
    #=========================================================================================#
    
    # Define the calib params #
    omega = calibVals[0]
    phi = calibVals[1]
    kappa = calibVals[2]
    XL = calibVals[3]
    YL = calibVals[4]
    ZL = calibVals[5]
    f = calibVals[6]
    x0 = calibVals[7]
    y0 = calibVals[8]
    
    # Set up the rotation matrix #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
    #=========================================================================================#    
    #2. Define all of the constants from known parameters- see derivation
    #=========================================================================================#
    c1 = x
    c2 = x0
    c3 = f
    c4 = m11
    c5 = m11*XL
    c6 = m12
    c7 = m12*YL
    c8 = m13*Z
    c9 = m13*ZL
    c10 = m31
    c11 = m31*XL
    c12 = m32
    c13 = m32*YL
    c14 = m33*Z
    c15 = m33*ZL
    d1 = (c1-c2)/(-c3)
    d2 = -c11-c13+c14-c15
    d3 = -c5-c7+c8-c9
    E = (d1*c10)-c4
    F = c6-(d1*c12)
    G = d3-(d1*d2)
    
    j1 = y
    j2 = y0
    j3 = m21
    j4 = m21*XL
    j5 = m22
    j6 = m22*YL
    j7 = m23*Z
    j8 = m23*ZL
    k1 = (j1-j2)/(-c3)
    k2 = -j4-j6+j7-j8
    L = (k1*c10)-j3
    M = j5-(k1*c12)
    N = k2-(k1*d2)   
    
    #=========================================================================================#    
    #3. Compute X and Y
    #=========================================================================================#    
    Y = (G-((E*N)/L))/(((E*M)/L)-F)
    X = ((F*Y)+G)/E
    
    return X,Y


def calcXYZ_WithDistortion(calibVals,x,y,Z):
    
    '''
    Calculate the real-world (XY) coordinates of a point given the image (xy) coordinates and the geometry solution/calibration parameters.
    To do so, the collinearity cinditions with distortion included are solved analytically. This is not excessively complicated but is messy. A derivation for the solution 
    using the terminology in this function can be found at: https://drive.google.com/file/d/1HrWPLsQnw-4p2_Hef9ilJNGUo2qJ0PIX/view?usp=sharing
    
    Inputs:
        calibVals (array or list): The calibration parameters (aka the geometry solution, but also including some intrinsics) in the order: omega, phi, kappa, camX, camY, camZ, f, x0, y0.
        x,y (float): Image coordinates of point
        Z (float): Real-world Z coordinate of point 
        
    Returns:
        X,Y: The real-world coordinates of the point based on your geometry solution and calibration values. 
        
    Created by:
        Matthew P. Conlin, University of Florida
        05/2020
    '''
    
    #=========================================================================================#    
    #1. Break out the calibration parameters and calculate the elements of the rotation matrix
    #=========================================================================================#
    
    # Define the calib params #
    omega = calibVals[0]
    phi = calibVals[1]
    kappa = calibVals[2]
    XL = calibVals[3]
    YL = calibVals[4]
    ZL = calibVals[5]
    f = calibVals[6]
    x0 = calibVals[7]
    y0 = calibVals[8]
    k1 = calibVals[9]
    p1 = calibVals[10]
    
    # Set up the rotation matrix #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
    #=========================================================================================#    
    #2. Define all of the constants from known parameters- see derivation
    #=========================================================================================#
    c1 = x
    c1_1 = -((x-x0)*(k1*(x**2+y**2)))
    c1_2 = -(p1*(2*(x-x0)**2+(x**2+y**2)))
    c1 = c1-c1_1-c1_2
    c2 = x0
    c3 = f
    c4 = m11
    c5 = m11*XL
    c6 = m12
    c7 = m12*YL
    c8 = m13*Z
    c9 = m13*ZL
    c10 = m31
    c11 = m31*XL
    c12 = m32
    c13 = m32*YL
    c14 = m33*Z
    c15 = m33*ZL
    d1 = (c1-c2)/(-c3)
    d2 = -c11-c13+c14-c15
    d3 = -c5-c7+c8-c9
    E = (d1*c10)-c4
    F = c6-(d1*c12)
    G = d3-(d1*d2)
    
    j1 = y
    j1_1 = -((y-y0)*(k1*(x**2+y**2)))
    j1_2 = -(2*p1*(x-x0)*(y-y0))
    j1 = j1-j1_1-j1_2
    j2 = y0
    j3 = m21
    j4 = m21*XL
    j5 = m22
    j6 = m22*YL
    j7 = m23*Z
    j8 = m23*ZL
    k1 = (j1-j2)/(-c3)
    k2 = -j4-j6+j7-j8
    L = (k1*c10)-j3
    M = j5-(k1*c12)
    N = k2-(k1*d2)   
    
    #=========================================================================================#    
    #3. Compute X and Y
    #=========================================================================================#    
    Y = (G-((E*N)/L))/(((E*M)/L)-F)
    X = ((F*Y)+G)/E
    
    return X,Y



def RectifyImage(calibVals,img,grd):
    
    '''
    Function to rectify an image using the resolved calibration parameters. User inputs a grid in real world space
    onto which the image is rectified.

    Inputs:
        calibVals: (array) The calibration vector returned by calibrate_PerformCalibration function
        img: (array) The image to be rectified
        grd: (list) Real-world grid onto which to planimetrically rectify the image in the order [xmin,xmax,dx,ymin,ymax,dy,z]

    Outputs:
        im_rectif: (array) The rectified image
        extents: (array) The geographic extents of the rectified image, for plotting purposes
        
    Created by:
        Matthew P. Conlin, University of Florida
        05/2020
    '''
    
    # Define the calib params #
    omega = calibVals[0]
    phi = calibVals[1]
    kappa = calibVals[2]
    XL = calibVals[3]
    YL = calibVals[4]
    ZL = calibVals[5]
    f = calibVals[6]
    x0 = calibVals[7]
    y0 = calibVals[8]
    
    # Set up the rotation matrix #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
    # Set up object-space grid #
    xg = np.arange(grd[0],grd[1],grd[2])
    yg = np.arange(grd[3],grd[4],grd[5])
    xgrd,ygrd = np.meshgrid(xg,yg)
    zgrd = np.zeros([len(xgrd[:,1]),len(xgrd[1,:])])+grd[6]
    extents = np.array([(-.5*grd[2])+min(xg),max(xg)+(.5*grd[2]),min(yg)-(.5*grd[5]),max(yg)+(.5*grd[5])])

    # Get image coordinates of each desired world coordinate based on calib vals #
    x = x0 - (f*(((m11*(xgrd-XL)) + (m12*(ygrd-YL)) + (m13*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))
    y = y0 - (f*(((m21*(xgrd-XL)) + (m22*(ygrd-YL)) + (m23*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))

    xx = x.flatten();yy = y.flatten()
    pts = list(zip(xx,yy))
    
    # Create photo coordinate vectors #
    u = np.arange(len(img[0,:,1]))
    v = np.arange(len(img[:,0,1]))

    # Create the interpolation functions #
    funcR = rgi((u,v),np.transpose(img[:,:,0]),bounds_error=False,fill_value=0)
    funcG = rgi((u,v),np.transpose(img[:,:,1]),bounds_error=False,fill_value=0)
    funcB = rgi((u,v),np.transpose(img[:,:,2]),bounds_error=False,fill_value=0)

    # Interpolate the color values for each channel #
    col_R = funcR(np.array(pts))
    col_G = funcG(np.array(pts))
    col_B = funcB(np.array(pts))

    # Reshape the color channel values to an image #
    col_R = np.reshape(col_R,[len(x[:,0]),len(x[0,:])])
    col_G = np.reshape(col_G,[len(x[:,0]),len(x[0,:])])
    col_B = np.reshape(col_B,[len(x[:,0]),len(x[0,:])])

    # Create therectified image #
    im_rectif = np.stack([col_R,col_G,col_B],axis=2)
    im_rectif = np.flipud(im_rectif)
    
    
    return im_rectif,extents



def calcCheckPointResid(calibVals,gcpxy,gcpFile,checks,camLoc):
    
    '''
    Calculate the residuals of check points (Euclidean distance) based on intrinsic/extrinsic calibration
    and point measurements (with e.g. GPS).
    
    Inputs:
        calibVals (list): A list containing the corrected calibration values in the order [omega,phi,kappa,camX,camY,camZ,f,x0,y0]
        gcpxy (array): nx2 array of image coordinates (U,V) for each of the points 
        gcpFile (str): path to the file containing the measured GCP locations
        checks (list): list containing the numbers of the GCPs to use as checkpoints (first point is number 1)
        camLoc (tuple): measured camera location (camX,camY)
        
    Returns:
        difs (array): nx2 array of residual in each direction, column 1 for the x-dir and column 2 for the y-dir
        resids (array): nx1 array of computed residuals for each point
        rms (float): The root mean squared value of computed residuals
        gcpXYreproj (array): nx2 array of reprojected positions for each point (X,Y)
        
    '''
    try:
        gcpFile = np.loadtxt(gcpFile,delimiter=' ')  
    except:
        gcpFile = np.loadtxt(gcpFile,delimiter=',') 
    camX = camLoc[0]
    camY = camLoc[1]
       
    #=============================================================================#
    # Calculate the reprojected positions of each GCP based on calibVals #
    #=============================================================================#   
    gcpXYreproj = np.empty([0,3])
    
    for i in checks:
        X,Y = calcXYZ(calibVals,gcpxy[i-1,0],gcpxy[i-1,1],gcpFile[i-1,3]) 
    
        X = X+camX
        Y = Y+camY  
        
        gcpXYreproj = np.vstack([gcpXYreproj,np.hstack([int(i),X,Y])])
        
        
    #=============================================================================#
    # Calculate the residuals #
    #=============================================================================#  
    difs = gcpFile[checks-1,1:3] - gcpXYreproj[:,1:3]
    resids = np.sqrt(difs[:,0]**2 + difs[:,1]**2) 
    rms = np.sqrt(np.sum(np.power(resids,2))/len(resids))
    
    return difs,resids,rms,gcpXYreproj
        
        
        
        
        
        
        
        
    
    
    


