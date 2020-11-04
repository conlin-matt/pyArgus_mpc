#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:39:43 2020

@author: matthewconlin
"""

# Standard library imports #
import math
import os

# Third-party imports #
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d

# Project imports #
from pyArgus_mpc import computation as comp, shorelineMapping as sl, utils
from pyArgus_mpc.SurfcamArgus import analysisTools as sca
from pyArgus_mpc.SurfcamArgus.GeophysicalAnalyses import vidPrep



#=============================================================================#
# Establish things we need later #
#=============================================================================#
vidDirec = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/RawVideoData/' # Directory for videos #

f = open('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/SensitivityTest/calibVals_optim.pkl','rb') # Get these from SurfCaT #
calibVals = pickle.load(f)

rectif_xmin = 150
rectif_xmax = 350
rectif_dx = .2
rectif_ymin = -200
rectif_ymax = -50
rectif_dy = .2

wlobj = utils.NOAAWaterLevelRecord(station=8722670,bdate='20200301',edate='20200801') # Water level record #
wl = wlobj.get()

#=============================================================================#
# Loop to create timex, rectify, and ID shoreline for each video we want to use #
#=============================================================================#
shorelines = {}
vids = ['202003051200','202008011500','202008021146'] # Chose these because they all have similar observed water levels #
h = None
for vid in vids:
    
    # Reduce the video #
    if not os.path.exists(vidDirec+'StLucie_'+vid+'_reduced.avi'):
        vidPrep.ReduceVid(vidDirec+'StLucie_'+vid+'.ts') # Reduce the vid to 1 frame per sec
     
    # Make the timex #    
    if not os.path.exists(vidDirec+'StLucie_'+vid+'_timex.png'):
        timex = sca.CreateImageProduct(vidDirec+'StLucie_'+vid+'_reduced.avi',1)
        
        timex_towrite = np.stack([timex[:,:,2]*255,timex[:,:,1]*255,timex[:,:,0]*255],axis=2)
        cv2.imwrite(vidDirec+'StLucie_'+vid+'_timex.png',timex_towrite) 
    else:
        timex = mpimg.imread(vidDirec+'StLucie_'+vid+'_timex.png')
        
    # Rectify the timex # # WL not available from API for August at time of writing #
    if vid == vids[1]:
        z_rectif = -0.5
    elif vid == vids[2]:
        z_rectif = -0.37
    else:
        z_rectif = wlobj.atTime(wl,vid)
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # ID the shoreline. Need to take a couple contours because of the trees #
    if h is not None:
        c,_h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif],h)
    else:
        c,h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
        
    if vid==vids[0]:
        xyz = np.vstack([c[0],c[1]]) # Note that the specific contours taken may need to change based on the bounding box that was selected #
    elif vid==vids[1]:
        xyz = np.vstack([c[0],c[2]])
    elif vid==vids[2]:
        xyz = np.vstack([c[0],c[1]])
        
    # Check the shoreline position visually #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(im_rectif,extent=extents,interpolation='bilinear')
    plt.axis('equal')
    plt.plot(xyz[:,0],xyz[:,1],'r')
    fig.show()
    plt.savefig(vidDirec+'StLucie_'+vid+'_rectif.png')
    
    # Save the shoreline #
    shorelines[vid] = xyz
 
      
   
#=============================================================================#
# Make the summary figure #
#=============================================================================#
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(6,4))
ax1 = plt.axes([.1,.7,.2,.2])
ax2 = plt.axes([.4,.7,.2,.2],sharey=ax1)
ax3 = plt.axes([.7,.7,.2,.2],sharey=ax1)
ax4 = plt.axes([.1,.1,.8,.5])
ax4.set_xlabel('Relative easting (m)',fontsize=10)
ax4.set_ylabel('Relative northing (m)',fontsize=10)

ax1.text(0.06, 0.97, 'a', transform=ax1.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax2.text(0.06, 0.97, 'b', transform=ax2.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax3.text(0.06, 0.97, 'c', transform=ax3.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax4.text(0.02, 0.97, 'd', transform=ax4.transAxes,fontsize=8, fontweight='bold', va='top',color='k')

l = []
for vid,sub,shoreline,col in zip(vids,[ax1,ax2,ax3],shorelines,['r','g','b']):
    
    timex = mpimg.imread(vidDirec+'StLucie_'+vid+'_timex.png')
   
    # Rectify the timex # # WL not available from API for August at time of writing #
    if vid == vids[1]:
        z_rectif = -0.5
    elif vid == vids[2]:
        z_rectif = -0.37
    else:
        z_rectif = wlobj.atTime(wl,vid)
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # Plot the timex with the shoreline ID #
    sub.imshow(im_rectif,extent=extents,interpolation='bilinear')
    sub.axis('equal')
    sub.plot(shorelines[shoreline][:,0],shorelines[shoreline][:,1],col)

    l.append(ax4.plot(shorelines[shoreline][:,0],shorelines[shoreline][:,1],col,linewidth=2))
    ax4.axis('equal')

l = [l[0][0],l[1][0],l[2][0]]
ax4.legend(l,('March 5 2020 ($\eta = -0.41$ m)','August 1 2020 ($\eta = -0.50$ m)','August 2 2020 ($\eta = -0.37$ m)'),loc='lower right')
ax4.arrow(135,-150,10,-2,head_width=3,fc='k')  
ax4.arrow(135,-150,-10,2,head_width=3,fc='k')  
ax4.text(146,-159,'Ocean')
ax4.text(115,-144,'Bay')



#=============================================================================#
# Analyze #
#=============================================================================# 
### Area change between March and Aug 1 ###
plt.figure()
plt.plot(shorelines['202003051200'][:,0],shorelines['202003051200'][:,1],'r')
plt.plot(shorelines['202008011500'][:,0],shorelines['202008011500'][:,1],'g')

# Calculate area of loss on southern side #
sl1 = shorelines['202003051200']; sl2 = shorelines['202008011500']
sl1 = sl1[np.where(np.logical_and(sl1[:,0]<270,sl1[:,1]<-118))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]<270,sl2[:,1]<-118))[0],:]

# Interpolate to even spacing #
xi = np.linspace(min(sl1[:,0]),max(sl1[:,0]),1500)
f1 = interp1d(sl1[:,0],sl1[:,1],bounds_error=False);yi1 = f1(xi)
f2 = interp1d(sl2[:,0],sl2[:,1],bounds_error=False);yi2 = f2(xi)

plt.figure()
plt.plot(sl1[:,0],sl1[:,1],'r')
plt.plot(sl2[:,0],sl2[:,1],'g')
plt.plot(xi,yi1,'r--')
plt.plot(xi,yi2,'g--')

# Area between the curves #
zdif = yi2-yi1
a_loss = np.trapz(zdif[~np.isnan(zdif)],xi[~np.isnan(zdif)],dx=xi[1]-xi[0])



# Now calculate area of gain on northern side #
sl1 = shorelines['202003051200']; sl2 = shorelines['202008011500']
sl1 = sl1[np.where(np.logical_and(sl1[:,0]<250,sl1[:,1]>-110))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]<250,sl2[:,1]>-110))[0],:]
# Rotate so we are looking mostly cross-shore #
theta = math.radians(-60)
R = np.vstack([np.hstack([math.cos(theta),-math.sin(theta)]),np.hstack([math.sin(theta),math.cos(theta)])])
sl1 = np.transpose(R@np.transpose(sl1));sl2 = np.transpose(R@np.transpose(sl2))
# Interpolate to even spacing #
xi = np.linspace(min(sl1[:,0]),max(sl1[:,0]),1500)
f1 = interp1d(sl1[:,0],sl1[:,1],bounds_error=False);yi1 = f1(xi)
f2 = interp1d(sl2[:,0],sl2[:,1],bounds_error=False);yi2 = f2(xi)
# Area between the curves #
zdif = yi2-yi1
a_gain = np.trapz(zdif[np.isnan(zdif)==False],xi[np.isnan(zdif)==False],dx=xi[1]-xi[0])






### Area change between Aug 1 and Aug 2 ###
sl1 = shorelines['202008011500']; sl2 = shorelines['202008021146']
sl1 = sl1[np.where(np.logical_and(sl1[:,0]>200,sl1[:,1]<-100))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]>200,sl2[:,1]<-100))[0],:]

# Interpolate to even spacing #
xi = np.linspace(min(sl1[:,0]),max(sl1[:,0]),1500)
f1 = interp1d(sl1[:,0],sl1[:,1],bounds_error=False);yi1 = f1(xi)
f2 = interp1d(sl2[:,0],sl2[:,1],bounds_error=False);yi2 = f2(xi)

# Area between the curves #
zdif = yi2-yi1
a_loss = np.trapz(zdif[0:1090],xi[0:1090],dx=xi[1]-xi[0])


