#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:21:12 2020

@author: matthewconlin
"""

# Standard library imports #
import math

# Third-party imports #
from matplotlib import path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy import interpolate,stats
from scipy.optimize import curve_fit
from skimage import filters,measure
from skimage.color import rgb2gray
from skimage.measure import profile_line
from sklearn.linear_model import LinearRegression






def mapShorelineSLIM(im_rectif,grd,transects=None):
    
    '''
    Function to map the shoreline on an image using the Shoreline Intensity Maximum (SLIM)
    technique first introduced by Plant and Holman, 1997 (Marine Geology). This function largely
    follows the algorithm laid out in Madsen and Plant, 2001 (Marine Geology). Noteable deviations
    from that algorithm include:
        - This function takes the simpler approach of using the maximum intensity along the profile as
        the initial guess for the SLIM location, rather than using a function of tide, time, and 
        alongshore distance from a bunch of manual identifications
        - This function does not establish a subwindow for the function fit based on the area under the curve. Rather,
        this function simply limits the profile to +/- 30 m cross-shore distance from the initial estimate. 
        This could (should?) probably be improved to use the subwindow approach.
        - This function uses somewhat less stringent criteria to accept/reject SLIM estimates. Since we do not establish
        the function of longshore distance, tide, and time (see point 1), we use only criteria 1-4 (of 6) presented 
        in Table 1 of Madesn and Plant, 2001 (Marine Geology)
        
    Inputs:
        im_rectif (array): a rectified image as a three page array (m x n x 3 array)
        grd (list): A list describing the real-world coordinates of the rectified product in the following format: [xmin,xmax,dx,ymin,ymax,dy,z]
        transects (list), optional: If you aleady have transects derived from the GetTransects function, you can pass them in this parameter
        
    Returns:
        slim (list): List of SLIM coordintes at each transect
        transects (list): the end points of the transects
        
    Note: I have commands to create figures at helpful visuaization points commented out, comment-in to create the figures 
        
        
    Created by:
    Matthew P. Conlin, University of Florida
    09/2020      
    '''
    
    # Identify the site curvature to get cross-shore transects #
    if not transects:
        transects = getTransects(im_rectif,grd,100)
    else:
        pass
    
    # Convert image to grayscale #
    im_rectif_g = rgb2gray(im_rectif)
    
    # Vectors of real-world coordinates #    
    xg = np.arange(grd[0]+.1,grd[1]+.1,grd[2])
    yg = np.arange(grd[3]+.1,grd[4]+.1,grd[5])
    yg = np.flipud(yg)
    
    # Get the SLIM at each transect #
    slim = []
    for t in transects:
        
        #============================================================#
        # Get the intensity profile along the transect and limit it to +-30 m from SLIM estimate #
        #============================================================#
        # This big line is finding the pixel coordinates closest to the transect's real-world coordinates, and formatting conveniently for later #
        t_im = np.array([[int(np.where(np.abs(xg-t[0,0]) == min(np.abs(xg-t[0,0])))[0]),int(np.where(np.abs(yg-t[0,1]) == min(np.abs(yg-t[0,1])))[0])],[int(np.where(np.abs(xg-t[1,0]) == min(np.abs(xg-t[1,0])))[0]),int(np.where(np.abs(yg-t[1,1]) == min(np.abs(yg-t[1,1])))[0])]])
        
        # Get the intensity profile along the transect #
        p1 = profile_line(im_rectif_g,np.flip(t_im[0]),np.flip(t_im[1]))
        
        # Discretize the transect #
        tx_world = np.linspace(t[0,0],t[1,0],len(p1))
        ty_world = np.linspace(t[0,1],t[1,1],len(p1))
        tdist_world = np.append(0,np.cumsum(np.sqrt(np.diff(tx_world)**2+np.diff(ty_world)**2)))
        dx = tdist_world[1]-tdist_world[0]

        # Find the closest transect point to the picked SLIM #
        iSLIM = int(np.where(p1==max(p1))[0][0])

        # Limit the intensity profile to plus or minus 30 m from the initial estimate #
        if iSLIM>30/dx:
            tdist_world = tdist_world[int(iSLIM-round(30/dx)):int(iSLIM+round(30/dx))]
            p1 = p1[int(iSLIM-round(30/dx)):int(iSLIM+round(30/dx))]
        else:
            tdist_world = tdist_world[0:int(iSLIM+round(30/dx))]
            p1 = p1[0:int(iSLIM+round(30/dx))]

        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        plt.imshow(im_rectif,interpolation='bilinear')
#        plt.axis('equal')
#        plt.plot(t_im[:,0],t_im[:,1],'r') 
#        plt.plot(tx,ty,'b')
#        plt.plot(tx[iSLIM],ty[iSLIM],'g*')
        
        
        #============================================================#
        # Normalize the limited intensity profile #
        #============================================================#
        pnorm = (p1-min(p1))/(max(p1)-min(p1))
       
#        fig = plt.figure()
#        plt.plot(tdist_world,pnorm)
        
        #============================================================#
        # Fit the profile with a quadratic #
        #============================================================# 
        def model(x,I0,I1,I2,Aslim,Xslim,Lslim):
            out = I0+(I1*x)+(I2*x**2)+(Aslim*np.exp(-(((x-Xslim)/Lslim)**2)))
            return out
        popt,pcov = curve_fit(model,tdist_world,pnorm,p0=[0.0001,0.0001,0.0001,0.1,float(tdist_world[pnorm==max(pnorm)][0]),5],maxfev = 100000)
        
#        fig = plt.figure()
#        plt.plot(tdist_world,pnorm,'b.')
#        plt.plot(tdist_world,model(tdist_world,*popt),'r-')
#        plt.plot(popt[4],1,'g*')
#        plt.draw()
#        plt.pause(0.001)
#        input('Enter to continue')
#        plt.close()
        

        #============================================================#
        # Take the SLIM location as the max of the quadratic #
        #============================================================#  
        slim_here = popt[4]
        
        #============================================================#
        # Get the x and y coords of the location (use similar triangles with transect end points) #
        #============================================================#          
        x = t[1][0]-t[0][0]
        y = t[1][1]-t[0][1]
        d = math.sqrt(x**2 + y**2)
        d_slim = slim_here
        
        x_slim = float((x*d_slim)/d) + t[0][0]
        y_slim = float((y*d_slim)/d) + t[0][1]
        
        #============================================================#
        # Save the SLIM location #
        #============================================================#          
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        plt.imshow(im_rectif,extent=extents,interpolation='bilinear')
#        plt.axis('equal')
#        plt.plot(x_slim,y_slim,'r*')
#        plt.show()
        
        if 0<popt[5]<25 and popt[4]>0:          
            slim.append(np.array([x_slim,y_slim]))
        else:
            slim.append(np.array([np.nan,np.nan]))
        
    return slim,transects




def mapShorelineCCD(im_rectif,grd,h=None):
    
    '''
    Functiion to map the shoreline on a geo-rectified image using the Color Channel Divergence (CCD) technique,
    described in Plant et al., 2007 (Journal Coastal Research). This code is essentially a translation
    of Mitch Harley's Matlab code available in the CIRN GitHub repos
    (https://github.com/Coastal-Imaging-Research-Network/Shoreline-Mapping-Toolbox/blob/master/mapShorelineCCD.m).
    
    The region of interest in which to perform the thresholding is identifed manually as four corners. 
    
    Inputs:
        im_rectif (array): a rectified image as a three page array (m x n x 3 array)
        grd (list): A list describing the real-world coordinates of the rectified product in the following format: [xmin,xmax,dx,ymin,ymax,dy,z]
        h (list), optional: The (four) verticies of the region of interest (as a list of tuples or lists) can be passed if you already have them.
        
    Returns:
        c (list): List of individual contours identified to be the shoreline, sorted by length. In general the 
                  desired shoreline will be the longest contour, i.e. c[0]
        h (list): The (four) verticies of the region of interest
                  
    
    Created by:
    Matthew P. Conlin, University of Florida
    09/2020     
    '''
    

    #=========================================================================================#
    # Display the rectified image and identify a region of interest by clicking four corners #
    #=========================================================================================#
    if h is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(im_rectif,extent=[grd[0],grd[1],grd[3],grd[4]],interpolation='bilinear')
        plt.axis('equal')
        fig.show()
        cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
        h = plt.ginput(n=4,show_clicks=True,timeout=0)
        h = np.array(h)

    #=========================================================================================#
    # Establish transects every 5 m in the y-direction within the ROI and get the intensity values along them #
    #=========================================================================================#
    # Vectors of real-world coordinates #
    xg = np.arange(grd[0],grd[1],grd[2])
    yg = np.arange(grd[3],grd[4],grd[5])
    yg = np.flipud(yg)
    
    # Establish the x and y coordinates for the endpoints of each transect #
    tranx_min = min(h[:,0])
    tranx_max = max(h[:,0])
    tranSpaceY = 5
    tranY = np.arange(min(h[:,1]),max(h[:,1]),tranSpaceY)

    # For each transect, get the intensity profile along it and put all the profiles into one vector #
    pAll = np.empty([1,3])
    for i in tranY:
        # Create the transect endpoints in real-world coordinates #
        t = np.array([[tranx_min,i],[tranx_max,i]])
        
        # This next big line is finding the pixel coordinates closest to the transect's real-world coordinates, and formatting conveniently for later #
        t_im = np.array([[int(np.where(np.abs(xg-t[0,0]) == min(np.abs(xg-t[0,0])))[0]),int(np.where(np.abs(yg-t[0,1]) == min(np.abs(yg-t[0,1])))[0])],[int(np.where(np.abs(xg-t[1,0]) == min(np.abs(xg-t[1,0])))[0]),int(np.where(np.abs(yg-t[0,1]) == min(np.abs(yg-t[0,1])))[0])]])
        
        # Get the intensity profile along the transect #
        p1 = profile_line(im_rectif,np.flip(t_im[0]),np.flip(t_im[1]))
        p = np.array([i for i in p1 if sum(i)!=0 and not np.isnan(sum(i))]) # Get rid of portions of transect in black regions of the image (values of 0 and/or nan) #
        
        # Add the transect's intensity values to the big variable #
        pAll = np.append(pAll,p,axis=0)
    pAll = pAll[1:len(pAll)-1,:]


    #=========================================================================================#
    # Get the threshold from the intensity values #
    #=========================================================================================#
    # Get the Python pdf object #
    kde = stats.gaussian_kde(pAll[:,0]-pAll[:,2])

    # Get the pdf values and locations the same was as Matlabs ksdensity function would #
    pdf_locs = np.linspace(min(pAll[:,0]-pAll[:,2]),max(pAll[:,0]-pAll[:,2]),100) # These locations are equivalent to the locations given by Matlabs ksdensity function
    pdf_values = kde(pdf_locs)

    # Get the Otsu threshold #
    thresh_otsu = filters.threshold_otsu(pAll[:,0]-pAll[:,2])

    # Skew the threshold value towards the positive (sand) peak #
    thresh_weightings = [1/3,2/3]
    iBelowThresh = np.where(pdf_locs<thresh_otsu)
    iAboveThresh = np.where(pdf_locs>thresh_otsu)
    iMinPeak = int(iBelowThresh[0][np.where(pdf_values[iBelowThresh] == max(pdf_values[iBelowThresh]))])
    iMaxPeak = int(iAboveThresh[0][np.where(pdf_values[iAboveThresh] == max(pdf_values[iAboveThresh]))])
    thresh = thresh_weightings[0]*pdf_locs[iMinPeak] + thresh_weightings[1]*pdf_locs[iMaxPeak]

#    # Plot the histogram and threshold #
#    fig = plt.figure()
#    plt.plot(pdf_locs,pdf_values,'b',linewidth=2)
#    plt.plot(pdf_locs[iMinPeak],pdf_values[iMinPeak],'ro')
#    plt.plot(pdf_locs[iMaxPeak],pdf_values[iMaxPeak],'ro')
#    plt.plot([thresh,thresh],[min(pdf_values),max(pdf_values)],'k--')


    #=========================================================================================#
    # Find the shoreline contour based on the threshold value #
    #=========================================================================================#
    # Mask data outside of ROI #
    im_dif = im_rectif[:,:,0]-im_rectif[:,:,2]
    X,Y = np.meshgrid(xg,yg)
    ROI = path.Path(np.vstack([h,h[0,:]]))
    inAr = ROI.contains_points(np.hstack([np.reshape(X,[np.size(X),1]),np.reshape(Y,[np.size(X),1])]))
    inAr = np.reshape(inAr,np.shape(X))
    im_dif[~inAr] = np.nan

    # Extract the threshold contour #
    c = measure.find_contours(im_dif,thresh)
    
    # Transform to real-world coordinates #
    it = -1
    for ar in c:
        it+=1
        x1 = [np.interp(i1[1],np.arange(0,len(xg)),xg) for i1 in ar]
        y1 = [np.interp(i1[0],np.arange(0,len(yg)),yg) for i1 in ar]
        xyz1 = np.transpose(np.array([x1,y1]))
        c[it] = xyz1
                
    # Sort the identified contorus by length- in general the shoreline contour will be the longest one (i.e. c[0])
    c = sorted(c,key=len,reverse=True)
    
    return c,h




def getTransects(im_rectif,grd,tranLen):
    
    '''
    Function to create shore-normal transects for a site by manually-identifying the geometry of the
    site on a geo-rectified image. Used by shoreline detection algorithms in this module.
    
    Inputs:
        im_rectif (array): a rectified image as a three page array (m x n x 3 array)
        grd (list): A list describing the real-world coordinates of the rectified product in the following format: [xmin,xmax,dx,ymin,ymax,dy,z]
        tranLen (float): The length of the created transects (in m), which are centered on the identified orientation line.
    Returns:
        transects (list): List of transect endpoints. Each entry is a 2x2 array with where columns are x/y and rows are points.
        
    Created by:
    Matthew P. Conlin, University of Florida
    09/2020     
    '''
    
    #=========================================================================================#
    # Display the rectified image and identify a line describing the orientation of the region of interest
    #=========================================================================================#
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(im_rectif,extent=[grd[0],grd[1],grd[3],grd[4]],interpolation='bilinear')
    plt.axis('equal')
    fig.show()
    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
    h = plt.ginput(n=0,show_clicks=True,timeout=0)
    h = np.array(h)

    #=========================================================================================#
    # Establish cross-shore transects every 5 m along the trend-line of the site
    #=========================================================================================#
    # Interpolate the line to fine spacing #
    f = interpolate.interp1d(h[:,1],h[:,0])
    yi = np.linspace(min(h[:,1]),max(h[:,1]),5000)
    xi = f(yi)
    
    # Distance of every point from the first point #
    dist = np.append(0,np.cumsum(np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)))
    
    # Every 5 m along the line create a cross-shore transect. Do this by walking the line and shooting perpendiculars every 5 m along it #
    transects = list()
    pStart = 0
    pNow = 0
    while pStart+5<len(dist):
        # Find the point closes to 5 m away from the current start point #
        pNow = int(np.where(np.abs(dist-(dist[pStart]+5)) == min(np.abs(dist-(dist[pStart]+5))))[0])
        
        # Get the angle of a regression line through points near this one #
        reg = LinearRegression()
        xPts = xi[pNow-int(round((pNow-pStart)/2)):pNow+int(round((pNow-pStart)/2))]
        yPts = yi[pNow-int(round((pNow-pStart)/2)):pNow+int(round((pNow-pStart)/2))]
        reg.fit(xPts.reshape(-1,1),yPts.reshape(-1,1))
        ang = reg.coef_
        
        # Create a transect passing through the center point of the regression line and perpendicular to it #
        ang_perp = -1/ang
        
        # Discretize the transect as points #
        xx = tranLen/2 * math.cos(ang_perp)
        yy = tranLen/2 * math.sin(ang_perp)
        
        # Add the transect as end points (columns = x,y; rows = start,end) #
        transects.append(np.hstack([np.vstack([xi[pNow]-xx,xi[pNow]+xx]),np.vstack([yi[pNow]-yy,yi[pNow]+yy])]))
        
        # Move to the next section #
        pStart = pNow
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(im_rectif,extent=[grd[0],grd[1],grd[3],grd[4]],interpolation='bilinear')
    plt.axis('equal')
    for i in transects:
        plt.plot(i[:,0],i[:,1],'r')
    fig.show()  
    
    return transects
