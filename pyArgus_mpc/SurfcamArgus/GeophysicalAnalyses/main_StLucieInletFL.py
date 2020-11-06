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
from matplotlib import colorbar, colors, image as mpimg, patches, pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.io import loadmat


# Project imports #
from pyArgus_mpc import computation as comp, shorelineMapping as sl, utils
from pyArgus_mpc.SurfcamArgus import analysisTools as sca
from pyArgus_mpc.SurfcamArgus.GeophysicalAnalyses import vidPrep



#=============================================================================#
# Establish things we need later #
#=============================================================================#
vidDirec = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/RawVideoData/' # Directory for videos #

rectif_xmin = 100
rectif_xmax = 350
rectif_dx = .2
rectif_ymin = -200
rectif_ymax = -50
rectif_dy = .2

wlobj = utils.NOAAWaterLevelRecord(station=8722670,bdate='20200301',edate='20200801') # Water level record #
wl = wlobj.get()


#=============================================================================#
# 1. Calibrate the camera using SurfRCaT
    
#=============================================================================#
# Refer to the SurfRCaT github repo and SoftwareX publication for details. This will
# result in a calibVals file which contains the solved-for calibration parameters.
# I will load this in below #
f = open('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/SensitivityTest/calibVals_optim.pkl','rb') # Get these from SurfCaT #
calibVals = pickle.load(f)

# Make a figure of the remote-GCPs used to complete the SurfRCaT calibration #
im = mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Extrinsic/GCP_25.png')
f = open('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/gcps_im.pkl','rb'); gcps_im = pickle.load(f)

fig = plt.figure(figsize=(4,3.5))
ax1 = plt.axes([.05,.3,.9,.4])
ax2 = plt.axes([.25,.05,.2,.2])
ax3 = plt.axes([.05,.05,.2,.2])
ax4 = plt.axes([.55,.05,.2,.2])
ax5 = plt.axes([.75,.05,.2,.2])
ax6 = plt.axes([.25,.75,.2,.2])
ax7 = plt.axes([.05,.75,.2,.2])
ax8 = plt.axes([.55,.75,.2,.2])
ax9 = plt.axes([.75,.75,.2,.2])
ax1.text(0.02, 0.97, 'a', transform=ax1.transAxes,fontsize=8, fontweight='bold', va='top',color='k')
ax2.text(0.02, 0.97, 'e1', transform=ax2.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax3.text(0.02, 0.97, 'e2', transform=ax3.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax4.text(0.02, 0.97, 'd1', transform=ax4.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax5.text(0.02, 0.94, 'd2', transform=ax5.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax6.text(0.02, 0.97, 'b1', transform=ax6.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax7.text(0.02, 0.97, 'b2', transform=ax7.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax8.text(0.02, 0.97, 'c1', transform=ax8.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax9.text(0.02, 0.97, 'c2', transform=ax9.transAxes,fontsize=8, fontweight='bold', va='top',color='w')
ax1.imshow(im)
ax1.set_xticks([])
ax1.set_yticks([])
for p in gcps_im:
    ax1.plot(p[0],p[1],'r+',markersize=6)
rect1 = patches.Rectangle((100,950),200,120,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
rect2 = patches.Rectangle((300,760),100,80,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
rect3 = patches.Rectangle((10,500),290,120,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
rect4 = patches.Rectangle((750,420),450,200,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
con1 = patches.ConnectionPatch(xyA=(200,1080),xyB=(100,950),coordsA="data",coordsB="data",axesA=ax1,axesB=ax2,color='k')
con2 = patches.ConnectionPatch(xyA=(350,840),xyB=(400,760),coordsA="data",coordsB="data",axesA=ax1,axesB=ax4,color='k')
con3 = patches.ConnectionPatch(xyA=(150,500),xyB=(0,620),coordsA="data",coordsB="data",axesA=ax1,axesB=ax6,color='k')
con4 = patches.ConnectionPatch(xyA=(975,420),xyB=(1200,620),coordsA="data",coordsB="data",axesA=ax1,axesB=ax8,color='k')
ax1.add_patch(rect1) 
ax1.add_patch(rect2)  
ax1.add_patch(rect3)  
ax1.add_patch(rect4)  
ax1.add_artist(con1)
ax1.add_artist(con2)
ax1.add_artist(con3)
ax1.add_artist(con4)
ax2.imshow(im)
ax2.set_xticks([])
ax2.set_yticks([])
for p in gcps_im:
    ax2.plot(p[0],p[1],'r+',markersize=10)
ax2.set_xlim(100,300)
ax2.set_ylim(1080,950)
ax3.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/GCPs_lidar_ex_1.png'))
ax3.set_xticks([])
ax3.set_yticks([])
ax3.plot(280.64872343565526,350.99125590318766,'k+')
ax3.plot(259.70885601141276,301.2250774793388,'k+')  
ax4.imshow(im)
ax4.set_xticks([])
ax4.set_yticks([])
for p in gcps_im:
    ax4.plot(p[0],p[1],'r+',markersize=10)
ax4.set_xlim(300,400)
ax4.set_ylim(840,760)
ax5.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/GCPs_lidar_ex_2.png'))
ax5.set_xticks([])
ax5.set_yticks([])
ax5.plot(370.8459452479338,316.83836334120423,'k+')  
ax6.imshow(im)
ax6.set_xticks([])
ax6.set_yticks([])
for p in gcps_im:
    ax6.plot(p[0],p[1],'r+',markersize=10)
ax6.set_xlim(0,300)
ax6.set_ylim(620,500)
ax7.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/GCPs_lidar_ex_3.png'))
ax7.set_xticks([])
ax7.set_yticks([])
ax7.plot(447.7851122542679,244.56445777346744,'k+')
ax8.imshow(im)
ax8.set_xticks([])
ax8.set_yticks([])
for p in gcps_im:
    ax8.plot(p[0],p[1],'r+',markersize=10)
ax8.set_xlim(750,1200)
ax8.set_ylim(620,420)
ax9.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Results_SurfRCaT/GCPs_lidar_ex_4.png'))
ax9.set_xticks([])
ax9.set_yticks([])
ax9.plot(623.6395073180788,290.35301999587693,'k+')



#=============================================================================#
# 2. Calculate and visualize the accuracy of the calibration by calculating checkpoint reprojection residuals

#=============================================================================#
gcpFile = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Extrinsic/gcpLocs.txt'
gcpxy = loadmat('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Extrinsic/allUV.mat')['UV']
checks = np.arange(1,32)
camLoc = (583381.79,3005482.72)
resids,rmsResid,gcpXYreproj = comp.calcCheckPointResid(calibVals,gcpxy,gcpFile,checks,camLoc)

# Make a figure showing the residuals on a rectified image #
im = mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/Extrinsic/GCP_25.png')
im_rectif,extents = comp.RectifyImage(calibVals,im,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,0])

scale_r = np.linspace(1,.5,11)
scale_g = np.linspace(1,0,11)
scale_b = np.linspace(1,0,11)
carr = np.transpose(np.vstack([scale_r,scale_g,scale_b]))
clist = [tuple(i) for i in carr]
cm = colors.ListedColormap(clist)
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(3,2.2))
ax1 = plt.axes([.2,.15,.7,.7])
cbax = plt.axes([.2,.81,.7,.03]) 
ax2 = plt.axes([.68,.215,.22,.3])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = plt.axes([.78,.23,.11,.25])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([])
ax1.imshow(im_rectif,extent=extents)
for i in range(0,len(gcpXYreproj)):
    r = np.floor(resids[i])
    if r>10:
        ci=10
    else:
        ci=int(r)
    ax1.plot(gcpXYreproj[i][1]-camLoc[0],gcpXYreproj[i][2]-camLoc[1],'o',markeredgecolor='k',markerfacecolor=clist[ci])
cb1=colorbar.ColorbarBase(cbax,cmap=cm,boundaries=[0,1,2,3,4,5,6,7,8,9,10,11],ticks=[0,1,2,3,4,5,6,7,8,9,10],spacing='proportional',orientation='horizontal',
                              label='Reprojection residual (m)',ticklocation='top')
ax1.set_xlabel('Relative Easting (m)',fontsize=8)
ax1.set_ylabel('Relative Northing (m)',fontsize=8)
ax1.set_xticks([100,150,200,250,300,350])
ax1.set_yticks([-50,-100,-150,-200])
ax3.boxplot(resids,sym='kx')
ax3.set_xticks([])
ax3.set_ylim(0,11)
ax3.set_xlim(0.9,1.1)
ax3.set_yticks([0,5,10])



#=============================================================================#
# 3. Perform the shoreline mapping analysis

#=============================================================================#

#=====================================================#
# 3.1. Loop to create timex, rectify, and ID shoreline for each video we want to use #
#=====================================================#
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
 
      
   
#============================================#
# 3.2. Make the summary figure #
#============================================#
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



#=========================================#
# 3.3. Analyze #
#=========================================#
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


