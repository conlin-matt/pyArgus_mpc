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
from datetime import datetime,timedelta
from matplotlib import colorbar, colors, image as mpimg, patches, pyplot as plt, dates as mdates
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression



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

wlobj2 = utils.NOAAWaterLevelRecord(8722670,'2020','2020') # Water level observations from one of two closest stations- Lake Worth #
wlobj1 = utils.NOAAWaterLevelRecord(8721604,'2020','2020') # Water level observations from other of two closest stations- Trident Pier #
wl1 = wlobj1.get()
wl2 = wlobj2.get()


#=============================================================================#
# 1. Calibrate the camera using SurfRCaT
    
#=============================================================================#
# Refer to the SurfRCaT github repo and SoftwareX publication for details. Running SurfRCaT
# results in a calibVals file which contains the solved-for calibration parameters.
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
difs,resids,rmsResid,gcpXYreproj = comp.calcCheckPointResid(calibVals,gcpxy,gcpFile,checks,camLoc)

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
fig = plt.figure(figsize=(4,3))
ax1 = plt.axes([.2,.15,.7,.7])
cbax = plt.axes([.2,.81,.7,.03]) 
ax2 = plt.axes([.6,.22,.39,.26])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = plt.axes([.69,.285,.06,.17])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([])
ax4 = plt.axes([.83,.285,.14,.17])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
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
ax3.set_xlim(0.85,1.15)
ax3.set_yticks([0,5,10])
ax3.set_xlabel('Residuals',fontsize=6)
ax4.plot(abs(difs[:,0]),abs(difs[:,1]),'.',markersize=2)
ax4.plot(np.linspace(0,10,20),np.linspace(0,10,20),'k-')
ax4.axis('equal')
ax4.set_xticks([0,10])
ax4.set_yticks([0,10])
ax4.set_xlabel('E-W',labelpad=-10,fontsize=6)
ax4.set_ylabel('N-S',labelpad=-15,fontsize=6)


#=============================================================================#
# 3. Perform the shoreline mapping analysis

#=============================================================================#

#=====================================================#
# 3.1. Loop to create timex, rectify, and ID shoreline for each video we want to use #
#=====================================================#
shorelines = {}
vids = ['202003051200','202008011500','202008021146'] # Chose these because they all have similar observed water levels #
h = None
zv = []
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
    z_rectif1 = wlobj1.atTime(wl1,vid) # Get the observed water level fromt the closest station at the top of the hour #
    z_rectif2 = wlobj2.atTime(wl2,vid) # Get the observed water level fromt the closest station at the top of the hour #
    distance_betweenStations = 215 # Along-coast distance (km) between the two stations
    distance_toInlet = 150 # Along-coast distance from northern station to Jupiter Inlet #
    z_rectif = np.interp(distance_toInlet,[0,distance_betweenStations],[z_rectif1,z_rectif2])
    zv.append(z_rectif)
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # ID the shoreline. Need to take a couple contours because of the trees #
    if h is not None:
        c,_h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif],h)
    else:
        c,h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
        
    if vid==vids[0]:
        xyz = np.vstack([c[0]]) # Note that the specific contours taken may need to change based on the bounding box that was selected #
    elif vid==vids[1]:
        xyz = np.vstack([c[0],c[1]])
    elif vid==vids[2]:
        xyz = np.vstack([c[0],c[1]])
        
    # Check the shoreline position visually #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(im_rectif,extent=extents,interpolation='bilinear')
    plt.axis('equal')
    plt.plot(xyz[:,0],xyz[:,1],'r')
    fig.show()
#    plt.savefig(vidDirec+'StLucie_'+vid+'_rectif.png')
    
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
ax4.text(0.97, 0.97, 'd', transform=ax4.transAxes,fontsize=8, fontweight='bold', va='top',color='k')

l = []
for vid,sub,shoreline,col,colu,a in zip(vids,[ax1,ax2,ax3],shorelines,['r','g','b'],[[240/255,128/255,128/255],[152/255,251/255,152/255],[135/255,206/255,250/255]],[.8,.6,.4]):
    
    timex = mpimg.imread(vidDirec+'StLucie_'+vid+'_timex.png')
   
    # Rectify the timex # # WL not available from API for August at time of writing #
    if vid == vids[1]:
        z_rectif = zv[0]
    elif vid == vids[2]:
        z_rectif = zv[1]
    else:
        z_rectif = zv[2]
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # Plot the timex with the shoreline ID #
    sub.imshow(im_rectif,extent=extents,interpolation='bilinear')
    sub.axis('equal')
    sub.plot(shorelines[shoreline][:,0],shorelines[shoreline][:,1],col)

#    # Produce uncertainty bounds #
#    if sub != ax2:
#        e = 3.59
#    else:
#        e = 3.59+1.3
#    v = np.vstack([shorelines[shoreline][:,0],shorelines[shoreline][:,1]]) 
#    ang = np.tan(np.divide(np.diff(v[1,:]),np.diff(v[0,:])))
#    ang_norm = ang+(math.pi/2)
#    mid = np.vstack([(np.diff(v[0,:])/2)+v[0,0:len(v[0,:])-1],(np.diff(v[1,:])/2)+v[1,0:len(v[0,:])-1]])
#    y1 = e*np.sin(ang_norm)+mid[1,:]
#    x1 = e*np.cos(ang_norm)+mid[0,:]
#    y2 = e*np.sin(-ang_norm)+mid[1,:]
#    x2 = e*np.cos(-ang_norm)+mid[0,:]   
#    
#    ax4.plot(x1,y1,color=colu,linewidth=10,alpha=a)
#    ax4.plot(x2,y2,color=colu,linewidth=10,alpha=a)
    
    l.append(ax4.plot(shorelines[shoreline][:,0],shorelines[shoreline][:,1],col,linewidth=2.5))
    ax4.axis('equal')

l = [l[0][0],l[1][0],l[2][0]]
ax4.legend(l,('March 5 2020 ($\eta = $'+str(round(zv[0],2))+' m)','August 1 2020 ($\eta = $'+str(round(zv[1],2))+' m)',
              'August 2 2020 ($\eta = $'+str(round(zv[2],2))+' m)'),loc='lower right',fontsize=6)
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
sl1 = sl1[np.where(np.logical_and(sl1[:,0]<270,sl1[:,0]>143))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]<270,sl2[:,0]>143))[0],:]
sl1 = sl1[sl1[:,1]<-118,:]; sl2 = sl2[sl2[:,1]<-118,:]
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
sl1 = sl1[np.where(np.logical_and(sl1[:,0]<250,sl1[:,1]>-118))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]<250,sl2[:,1]>-118))[0],:]

polygon = np.vstack([sl1,np.flipud(sl2)])
x = polygon[:,0];y = polygon[:,1]
a_gain = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



### Area change between Aug 1 and Aug 2 ###
sl1 = shorelines['202008011500']; sl2 = shorelines['202008021146']
sl1 = sl1[np.where(np.logical_and(sl1[:,0]>200,sl1[:,1]<-100))[0],:]; sl2 = sl2[np.where(np.logical_and(sl2[:,0]>200,sl2[:,1]<-100))[0],:]

# Rotate #
theta = math.radians(-22)
R = np.vstack([np.hstack([math.cos(theta),-math.sin(theta)]),np.hstack([math.sin(theta),math.cos(theta)])])
sl1 = np.transpose(R@np.transpose(sl1));sl2 = np.transpose(R@np.transpose(sl2))

# Interpolate to even spacing #
xi = np.linspace(min(sl1[:,0]),max(sl1[:,0]),1500)
f1 = interp1d(sl1[:,0],sl1[:,1],bounds_error=False);yi1 = f1(xi)
f2 = interp1d(sl2[:,0],sl2[:,1],bounds_error=False);yi2 = f2(xi)

# Area between the curves #
zdif = yi2-yi1
a_loss = np.trapz(zdif[0:1090],xi[0:1090],dx=xi[1]-xi[0])




#=========================================#
# 3.4. Look at outlines between March and August #
#=========================================#
shorelines2 = {}
vids = ['202003051200','202004301009','202005141050','202008011500','202008021146'] # Chose these because they all have similar observed water levels #
h = None
zv2 = []
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
    z_rectif1 = wlobj1.atTime(wl1,vid) # Get the observed water level fromt the closest station at the top of the hour #
    z_rectif2 = wlobj2.atTime(wl2,vid) # Get the observed water level fromt the closest station at the top of the hour #
    distance_betweenStations = 215 # Along-coast distance (km) between the two stations
    distance_toInlet = 150 # Along-coast distance from northern station to Jupiter Inlet #
    z_rectif = np.interp(distance_toInlet,[0,distance_betweenStations],[z_rectif1,z_rectif2])
    zv2.append(z_rectif)
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # ID the shoreline. Need to take a couple contours because of the trees #
    if h is not None:
        c,_h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif],h)
    else:
        c,h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
        
    if vid==vids[0]:
        xyz = np.vstack([c[0]]) # Note that the specific contours taken may need to change based on the bounding box that was selected #
    elif vid==vids[1]:
        xyz = np.vstack([c[0],c[1]])
    elif vid==vids[2]:
        xyz = np.vstack([c[0],c[1]])
    elif vid==vids[3]:
        xyz = np.vstack([c[0],c[1]])
    elif vid==vids[4]:
        xyz = np.vstack([c[0],c[1]])
        
    # Check the shoreline position visually #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(im_rectif,extent=extents,interpolation='bilinear')
    plt.axis('equal')
    plt.plot(xyz[:,0],xyz[:,1],'r')
    fig.show()
#    plt.savefig(vidDirec+'StLucie_'+vid+'_rectif.png')
    
    # Save the shoreline #
    shorelines2[vid] = xyz
    
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(6,4))
ax = fig.subplots(1)
ax.set_xlabel('Relative easting (m)',fontsize=10)
ax.set_ylabel('Relative northing (m)',fontsize=10)

l = []
for vid,shoreline,col in zip(vids,shorelines2,['r','g','b','m','y']):
    
    timex = mpimg.imread(vidDirec+'StLucie_'+vid+'_timex.png')
   
    # Rectify the timex # # WL not available from API for August at time of writing #
    if vid == vids[0]:
        z_rectif = zv2[0]
    elif vid == vids[1]:
        z_rectif = zv2[1]
    elif vid == vids[2]:
        z_rectif = zv2[2]
    elif vid == vids[3]:
        z_rectif = zv2[3]
    else:
        z_rectif = zv2[4]
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    
    l.append(ax.plot(shorelines2[shoreline][:,0],shorelines2[shoreline][:,1],col,linewidth=2.5))
    ax.axis('equal')

l = [l[0][0],l[1][0],l[2][0],l[3][0],l[4][0]]
ax.legend(l,('March 5 2020 ($\eta = $'+str(round(zv2[0],2))+' m)','April 30 2020 ($\eta = $'+str(round(zv2[1],2))+' m)', 'May 14 2020 ($\eta = $'+str(round(zv2[2],2))+' m)',
              'August 1 2020 ($\eta = $'+str(round(zv2[3],2))+' m)','August 2 2020 ($\eta = $'+str(round(zv2[4],2))+' m)'),loc='lower right',fontsize=6)
ax.arrow(135,-150,10,-2,head_width=3,fc='k')  
ax.arrow(135,-150,-10,2,head_width=3,fc='k')  
ax.text(146,-159,'Ocean')
ax.text(115,-144,'Bay')

area = []
for shoreline in shorelines2:
    polygon = shorelines2[shoreline]
    x = polygon[:,0]
    y = polygon[:,1]
    a = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    area.append(a)



#=========================================#
# Map the feature every day in May 2020
#=========================================#
direc = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/RawVideoData/2020-05/'
zv3 = []
f = open('/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/StLucie_area_20210208/CCD_extents.pkl','rb');h = pickle.load(f)
shorelines3 = {}
ii = -1
for vid in sorted([i for i in os.listdir(direc) if '.ts' in i or '.avi' in i]):
    
    ii+=1
    date = vid.split('_')[1].split('.')[0]
    
    # Reduce the video #
    if not os.path.exists(direc+'StLucie_'+date+'_reduced.avi'):
        vidPrep.ReduceVid(direc+'StLucie_'+date+'.ts') # Reduce the vid to 1 frame per sec
     
    # Make the timex #    
    if not os.path.exists(direc+'StLucie_'+date+'_timex.png'):
        timex = sca.CreateImageProduct(direc+'StLucie_'+date+'_reduced.avi',1)
        
        timex_towrite = np.stack([timex[:,:,2]*255,timex[:,:,1]*255,timex[:,:,0]*255],axis=2)
        cv2.imwrite(direc+'StLucie_'+date+'_timex.png',timex_towrite) 
    else:
        timex = mpimg.imread(direc+'StLucie_'+date+'_timex.png')
        
    # Rectify the timex # # WL not available from API for August at time of writing #
    z_rectif1 = wlobj1.atTime(wl1,date) # Get the observed water level fromt the closest station at the top of the hour #
    z_rectif2 = wlobj2.atTime(wl2,date) # Get the observed water level fromt the closest station at the top of the hour #
    distance_betweenStations = 215 # Along-coast distance (km) between the two stations
    distance_toInlet = 150 # Along-coast distance from northern station to St. Lucie Inlet #
    z_rectif = np.interp(distance_toInlet,[0,distance_betweenStations],[z_rectif1,z_rectif2])
    zv3.append(z_rectif)
           
    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # ID the shoreline. Need to take a couple contours because of the trees #
    if h is not None:
        c,_h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif],h)
    else:
        c,h = sl.mapShorelineCCD(im_rectif,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
    
    # Pull the correct contours for each image- varies a bit with image, I had to figure these out manually #
    if ii == 0:
        xyz = np.vstack([c[0]])
    elif ii == 2:
        xyz = np.vstack([c[0],c[2]])
    elif ii==3 or ii==19 or ii==20 or ii==26 or ii==27:
        xyz = np.vstack([c[1],c[0]])
    else:
        xyz = np.vstack([c[0],c[1]])
        
    shorelines3[date] = xyz



plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(6.5,7))
ax = fig.subplots(9,4)  
it = -1
for timex,shoreline,axisRow,axisCol in zip(sorted([i for i in os.listdir(direc) if 'timex' in i]),shorelines3,[0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8],
                                           [0,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,3]):
    
    it += 1
    timex1 = timex
    timex = mpimg.imread(direc+timex)
   
    # Rectify the timex # # WL not available from API for August at time of writing #
    z_rectif = zv3[it]
                   

    im_rectif,extents = comp.RectifyImage(calibVals,timex,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,z_rectif])
     
    ax[axisRow][axisCol].imshow(im_rectif,extent=extents,interpolation='bilinear')

    ax[axisRow][axisCol].plot(shorelines3[shoreline][:,0],shorelines3[shoreline][:,1],color='k',linewidth=2)
    ax[axisRow][axisCol].axis('equal')
    ax[axisRow][axisCol].text(150,-185,'$\eta = '+str(round(zv3[it],2))+'$',fontsize=6,color='w')
    dt = datetime.strptime(timex1.split('_')[1],'%Y%m%d%H%M')
    ax[axisRow][axisCol].text(75,-175,datetime.strftime(dt,'%b %d'),rotation=90,fontsize=6)
    
    if axisRow == 7 and axisCol==0:
        pass
    else:
         ax[axisRow][axisCol].set_xticks([])
         ax[axisRow][axisCol].set_yticks([])   
        
ax[0][1].axis('off')
ax[0][1].set_xlim(-200,100)
ax[0][1].set_ylim(-200,100)
ax[0][1].arrow(135,-150,10,-2,head_width=3,fc='k')  
ax[0][1].arrow(135,-150,-10,2,head_width=3,fc='k')  
ax[0][2].text(146,-159,'Ocean')
ax[0][2].text(115,-144,'Bay')
ax[0][2].axis('off')     
ax[0][3].axis('off')  
ax[8][0].axis('off')
ax[8][1].axis('off')
ax[8][2].axis('off')
ax[7][0].set_xlabel('Relative Easting (m)',fontsize=8)
ax[7][0].set_ylabel('Relative Northing (m)',fontsize=8)

# Calculate the area of the feature each day
e_total = 6.2
e_x = np.linspace(0,e_total,1000)
e_y = np.sqrt((e_total**2)-(e_x**2))

area_e = np.empty([0,len(shorelines3.keys())])
for ii in range(0,1000):
    area = []
    it = -1
    for key in shorelines3.keys():
        it+=1
        f = shorelines3[key]
        x = f[:,0]
        y = f[:,1]
        
        # Randomly vary each point's potion within its uncertainty #
        i = [random.randint(0,len(e_x)-1) for r in range(0,len(x))]
        x = x+e_x[i]
        y = y+e_y[i]
        
        # Correct for water level by shifting the outline everywhere in the cross shore to a reference tidal value of 0 m #
        z_tide = zv3[it]
        z_ref = 0
        m = 0.1
        d = (z_ref-z_tide)/m    
        xyz_new = np.empty([0,2])
        for ii in range(1,len(x)-1):
            theta = math.atan2(y[ii+1]-y[ii-1],x[ii+1]-x[ii-1])
    #        plt.plot(x,y,'b.')
    #        plt.plot(x[ii-1:ii+2],y[ii-1:ii+2],'r')
            theta_norm = theta+(math.pi/2)
            x_n = d*math.cos(theta_norm)+x[ii]
            y_n = d*math.sin(theta_norm)+y[ii]
    #        plt.plot(x_n,y_n,'g.')
            xyz_new = np.vstack([xyz_new,(x_n,y_n)])
        
        xyz_new_filt = savgol_filter(xyz_new[:,1],51,1)
    #    plt.plot(x,y,'b')
    #    plt.plot(xyz_new[:,0],xyz_new_filt,'r')        
        
        x = xyz_new[:,0]   
        y = xyz_new_filt
        a = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        area.append(a)
    area_e = np.vstack([area_e,area])
dtime_area = [datetime.strptime(i,'%Y%m%d%H%M') for i in list(shorelines3.keys())]   
    
fig,ax = plt.subplots(2,figsize=(6.5,3))
for i in range(0,len(shorelines3.keys())):
    ax[0].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i]-timedelta(hours=10),dtime_area[i]+timedelta(hours=10)),(np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i]-timedelta(hours=10),dtime_area[i]+timedelta(hours=10)),(np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
ax[0].plot(dtime_area[0],np.mean(area_e,axis=0)[0],'ks')
ax[0].plot(dtime_area[1:-1],np.mean(area_e,axis=0)[1:-1],'ks-')
ax[0].plot(dtime_area[-1],np.mean(area_e,axis=0)[-1],'ks')
for i in range(0,len(shorelines3.keys())):
    ax[1].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[1].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
    ax[1].plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[1].plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
ax[1].plot(dtime_area[1:-1],np.mean(area_e,axis=0)[1:-1],'ks-')
ax[1].set_xlim(dtime_area[1]-timedelta(days=1),dtime_area[-2]+timedelta(days=1))
ax[1].set_xticks(dtime_area[1:-1:5])
ax[0].plot((dtime_area[1]-timedelta(days=2),dtime_area[1]-timedelta(days=2)),(6100,7900),color=[.6,.6,.6])
ax[0].plot((dtime_area[-2]+timedelta(days=2),dtime_area[-2]+timedelta(days=2)),(6100,7900),color=[.6,.6,.6])
ax[0].plot((dtime_area[1]-timedelta(days=2),dtime_area[-2]+timedelta(days=2)),(6100,6100),color=[.6,.6,.6])
ax[0].plot((dtime_area[1]-timedelta(days=2),dtime_area[-2]+timedelta(days=2)),(7900,7900),color=[.6,.6,.6])
#con = ConnectionPatch(xyA=(dtime_area[1]-timedelta(days=2),6200), xyB=(datetime(2020,5,1),8000), coordsA="data", coordsB="data",
#                      axesA=ax[0], axesB=ax[1], color=[.6,.6,.6])
#ax[0].add_artist(con)
#con1 = ConnectionPatch(xyA=(dtime_area[-2]+timedelta(days=2),6100), xyB=(datetime(2020,6,1),8000), coordsA="data", coordsB="data",
#                      axesA=ax[0], axesB=ax[1], color=[.6,.6,.6])
#ax[0].add_artist(con1)
ax[0].set_ylabel('Area $(m^2)$')
ax[1].set_ylabel('Area $(m^2)$')
ax[0].set_ylim(6000,8000); ax[0].set_yticks([6000,6500,7000,7500])
ax[1].set_ylim(6000,8000); ax[1].set_yticks([6000,6500,7000,7500])
ax[0].text(0.01, 0.97, 'a', transform=ax[0].transAxes,fontsize=8, fontweight='bold', va='top',color='k')
ax[1].text(0.01, 0.97, 'b (inset)', transform=ax[1].transAxes,fontsize=8, fontweight='bold', va='top',color='k')

area = np.mean(area_e,axis=0)  





       
waveobj = utils.NDBCWaveRecord(41114,2020) # Wave observstions from closest buoy #
waves = waveobj.download()  
dt = []
for ii in range(0,len(waves)):
    i = waves.iloc[ii]
    dt.append(datetime(int(i['yr']),int(i['mo']),int(i['day']),int(i['hr']),int(i['mm'])))
dtime_waves =[]   
for i in dt: # Convert from UTC to local time, accounting for daylight savings #
    if i<datetime(2020,3,8,7,0) or i>=datetime(2020,11,1,6,0):
        hr_to_subtract = 5
    else:
        hr_to_subtract = 4
    dtime_waves.append(i-timedelta(hours=hr_to_subtract))
i = waves['MWD (degT)'] == 999
waves['MWD (degT)'][i] = np.nan
i = waves['MWD (degT)'] == 9999
waves['MWD (degT)'][i] = np.nan
i = waves['wvht (m)'] == 99
waves['wvht (m)'][i] = np.nan
i = waves['DPD (sec)'] == 999
waves['DPD (sec)'][i] = np.nan

  

dtime_wl= list(wl1['Time'])
dt = []
for i in dtime_wl:
    dt.append(datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[14:16])))
dtime_wl = dt

#R2_all = np.empty([38,0])
#for iii in range(0,100):
#    R2 = []
#    for dur in np.arange(2,90):
#        Hs = []
#        wl = []
#        w_cor = []
#        V = []
#        for w,t,n in zip(area[1:-1],dtime_area[1:-1],zv3[1:-1]):
#            dt_start_waves = t-timedelta(hours=int(dur))
#            dt_start_wl = t-timedelta(hours=int(dur))
#            dt_end = t
#            
#            # Waves #
#            i = 0
#            val = dtime_waves[0]
#            while val<dt_start_waves:
#                i+=1
#                val = dtime_waves[i]
#            i_start = i
#            
#            i = 0
#            val = dtime_waves[0]
#            while val<dt_end:
#                i+=1
#                val = dtime_waves[i]
#            i_end = i-1
#            
#            Hs.append(np.mean(np.array(waves.iloc[i_start:i_end]['wvht (m)'])))
#            ###
#            
#            # Water level #
#            i = 0
#            val = dtime_wl[0]
#            while val<dt_start_wl:
#                i+=1
#                val = dtime_wl[i]
#            i_start = i
#            
#            i = 0
#            val = dtime_wl[0]
#            while val<dt_end:
#                i+=1
#                val = dtime_wl[i]
#            i_end = i-1        
#            
#            wl.append(max(np.array(wl1.iloc[i_start:i_end]['wl_obs']))-min(np.array(wl1.iloc[i_start:i_end]['wl_obs'])))
#            V95 = 0.9
#            V45 = 0.4
#            c = abs(((max(np.array(wl1.iloc[i_start:i_end]['wl_obs']))-min(np.array(wl1.iloc[i_start:i_end]['wl_obs'])))/0.829))*100
#            V.append(((c-45)/45)*(V95-V45))
#            ###
#    
#            a_std = float(np.std(area_e,axis=0)[area==w])
#            a_change = random.choice(np.linspace(w-(2*a_std),w+(2*a_std),1000))
#            w_cor.append(a_change)
#        
#        reg = LinearRegression().fit(np.hstack([np.array(Hs).reshape(-1,1),np.array(V).reshape(-1,1)]),np.array(w_cor).reshape(-1,1))
#        yhat = reg.predict(np.hstack([np.array(Hs).reshape(-1,1),np.array(V).reshape(-1,1)]))
#        sst = np.sum((w_cor-np.mean(w_cor))**2)
#        ssr = np.sum((yhat-np.mean(w_cor))**2)
#        coef = (ssr/sst)
#        R2.append(coef)
#    R2_all = np.hstack([R2_all,np.array(R2).reshape(-1,1)])









def shoalWave(d,H,T,theta):
    '''
    Function to propagate a wave to breakpoint, assuming parallel and linear contours, using linear wave theory.
    
    args:
        d: The depth at which wave observations are collected (e.g. depth of buoy)
        H: Wave heiight measured at buoy
        T: Wave period measured at buoy
        theta: Wave direction measured at buoy
        
    returns:
        Hb: The breaking wave height
        dir_b: The breaking wave angle relative to the coast
    
    '''
    
    d_vec = np.arange(-d,0,0.2) # Vector of water depths #
    # Wave parameters at buoy #  
    k0 = utils.newtRaph(T,d)
    C0 = ((2*math.pi)/k0)/T
    n0 = 0.5*(1+((2*k0*d)/math.sinh(2*k0*d)))
    alpha0 = theta
    Hh = H
    ###
    # Calc new parameters at each depth #
    for h in d_vec[1:len(d_vec)]:
        k = utils.newtRaph(T,-h)
        C = ((2*math.pi)/k)/T
        n = 0.5*(1+((2*k*-h)/math.sinh(2*k*-h))) 
        alpha = math.degrees(math.asin((C/C0)*math.sin(math.radians(alpha0))))
        Hs = math.sqrt((n0*C0)/(n*C))*math.sqrt(math.cos(math.radians(alpha0))/math.cos(math.radians(alpha)))*Hh
        if Hs/-h>=0.78:
            Hb = Hs
            dir_b = alpha
            break
        else:
            k0 = k
            C0 = C
            n0 = n
            alpha0 = alpha
            Hh = Hs
    return Hb,dir_b


def calcAveragedCurrents(start_time,averaging_time_waves,averaging_time_wl,timeVec_waves,timeVec_wl,valueVec_waves,valueVec_wl):
    '''
    Function to calculate time-averaged longshore current and tidal current velocity for specified length of time
    prior to specified time. This is to create a regression model between these parameters and visible sand body area.
    
    Longshore current vel. is calculated using Equation 8.14b in Komar (1998). Waves measured at Buoy 41114 are first
    propagated to the breakpoint, assuming parallel and uniform contours, using linear wave theory.
    
    Tidal current vel is calculated following Tawil et al. (2019) J. Mar. Sci. and Eng, which uses observed tidal amplitude
    each tidal cycle relative to the mean amplitude, along with assumed spring and neap velocities, to estimate a velocity.
    
    args:
        start_time: Averaging will occur over some period of time prior to this date.
        averaging_time_waves: The amount of time before start_date for the longshore current velocity average to be calculated (in hours)
        averaging_time_wl: The amount of time before start_date for the tidal current velocity average to be calculated (in hours)
        timeVec_waves: The list of datetimes of wave observations
        timeVec_wl: The list of datetimes of water level observations
        valueVec_waves: The waves DataFrame generated from the call to utils.NDBCWaveRecord earlier in this script
        valueVec_wl: The wl DataFrame generated from the call to utils.NOAAWaterLevelRecord earlier in this script
    returns:
        Vl: The average longshore current velocity (m/s)
        Vt: The average tidal current velocity (m/s)
        
    '''
    
    dt_start_waves = start_time-timedelta(hours=int(averaging_time_waves))
    dt_start_wl = start_time-timedelta(hours=int(averaging_time_wl))
    dt_end = start_time
    
    ### Longshore current velocity ###
    # Determine start datetime #
    i = 0
    val = timeVec_waves[0]
    while val<dt_start_waves:
        i+=1
        val = timeVec_waves[i]
    i_start_waves = i
    # Determine end datetime #
    i = 0
    val = timeVec_waves[0]
    while val<dt_end:
        i+=1
        val = timeVec_waves[i]
    i_end_waves = i-1
    # Determine time averaged longshore current velocity before the start time #
    waves_partial = np.array(valueVec_waves.iloc[i_start_waves:i_end_waves])
    # Make sure the extracted portion of data actually spans the desired length of time. There is missing data in 2020.
    if len(waves_partial)==0 or datetime(dt_start_waves.year,dt_start_waves.month,dt_start_waves.day) != datetime(int(waves_partial[0,0]),int(waves_partial[0,1]),int(waves_partial[0,2])):
        Vl = np.nan
    else:
        Hb = []
        dir_b = []
        for i in waves_partial:
            d1 = i[-1]
            if d1<90:
                d = 90-d1-20 # -20 to correct for orientation of coastline #
            elif d1>90 and d1<180:
                d = d1-90-20
            else:
                d = np.nan
            if d is not np.nan:
                Hb1,dir_b1 = shoalWave(16.5,i[5],i[6],d)
            else:
                Hb1 = np.nan
                dir_b1 = np.nan
            Hb.append(Hb1)
            dir_b.append(dir_b1)
        V1 = np.sqrt(np.multiply(9.81,Hb))*np.sin(np.radians(dir_b))*np.cos(np.radians(dir_b))
        Vl = np.nanmean(V1)
        ########################################
    
    ### Water level ####
    # Determine start datetime #
    i = 0
    val = timeVec_wl[0]
    while val<dt_start_wl:
        i+=1
        val = timeVec_wl[i]
    i_start_wl = i
    # Determine end datetime #
    i = 0
    val = timeVec_wl[0]
    while val<dt_end:
        i+=1
        val = timeVec_wl[i]
    i_end_wl = i-1 
    # Determine time-averaged tidal current velocity before start time #
    V95 = 0.9
    V45 = 0.4
    p = find_peaks(np.array(valueVec_wl.iloc[i_start_wl:i_end_wl]['wl_obs']))
    V1 = []
    for ii in range(0,len(p[0])):
        wl_slice = np.array(valueVec_wl.iloc[i_start_wl:i_end_wl]['wl_obs'])[p[0][ii]:p[0][ii]+7]
        tide_range = np.max(wl_slice)-np.min(wl_slice)
        c = tide_range/0.829*100
        V1.append(((c-45)/45)*(V95-V45))
    Vt = np.nanmean(V1)
    ########################################
    
    return Vl,Vt

# Create the regression model for different averaging times and save R2 of each to find optimal #
R2_new = np.empty([0,4]) 
for dur_waves in np.arange(24,30*24,24):
    for dur_wl in np.arange(24,30*24,24):            
        Vl = []
        Vt = []
        w_cor = []
        a = area#[1:-1]
        for t in dtime_area:#[1:-1]:
            
            Vl1,Vt1 = calcAveragedCurrents(t,dur_waves,dur_wl,dtime_waves,dtime_wl,waves,wl1)
            Vl.append(Vl1)
            Vt.append(Vt1)
            
        if len(np.array(Vl)[~np.isnan(Vl)])<20:
            R2_new = np.vstack([R2_new,np.hstack([dur_waves,dur_wl,np.nan,np.nan])])
        else:
            
            try:
    #            reg = LinearRegression().fit(np.hstack([np.array(Vl).reshape(-1,1),np.array(Vt).reshape(-1,1)]),np.array(a).reshape(-1,1))
                reg = LinearRegression().fit(np.hstack([np.array(Vl)[~np.isnan(Vl)].reshape(-1,1),np.array(Vt)[~np.isnan(Vl)].reshape(-1,1)]),np.array(a)[~np.isnan(Vl)].reshape(-1,1))
            except:
                R2_new = np.vstack([R2_new,np.hstack([dur_waves,dur_wl,np.nan,np.nan])])
            else:
                yhat = reg.predict(np.hstack([np.array(Vl)[~np.isnan(Vl)].reshape(-1,1),np.array(Vt)[~np.isnan(Vl)].reshape(-1,1)]))
                sst = np.sum((a[~np.isnan(Vl)]-np.mean(a[~np.isnan(Vl)]))**2)
                ssr = np.sum((yhat-np.mean(a[~np.isnan(Vl)]))**2)
                coef = (ssr/sst)
                rmse = np.sqrt(np.sum((yhat-a[~np.isnan(Vl)])**2)/len(yhat))
                R2_new = np.vstack([R2_new,np.hstack([dur_waves,dur_wl,coef,rmse])])


# Visualize the results #
durWave_grd,durWL_grd = np.meshgrid(R2_new[:,0],R2_new[:,1])
R2_grd = griddata(R2_new[:,0:2],R2_new[:,2],(durWave_grd,durWL_grd))
fig = plt.figure(figsize=(3.5,3.1))
ax = plt.axes([.15,.15,.8,.7])
cbax = plt.axes([.15,.86,.8,.02])
h = ax.contourf(durWave_grd/24,durWL_grd/24,R2_grd,[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],cmap='hot')
ax.set_xlim([5,29])
ax.set_ylim([5,29])
ax.set_xlabel('$V_l$ averaging time (days)')
ax.set_ylabel('$V_t$ averaging time (days)')
fig.colorbar(h,cax=cbax,orientation='horizontal',ticklocation='top',label='$R^2$')
ax.plot(12,14,'k*',markersize=10)
ax.plot((0,12),(14,14),'k--')
ax.plot((12,12),(0,14),'k--')
r = Rectangle((13,1),17,30,color='w',alpha=.7)
ax.add_artist(r)

# Run the model that uses the optimal averaging times #
vals_best = R2_new[np.where(R2_new[:,2]==np.nanmax(R2_new[:,2]))[0],:]
dur_waves_best = int(vals_best[0][0])
dur_wl_best = int(vals_best[0][1])
Vl = []
Vt = []
a = area#[1:-1]
for t in dtime_area:#[1:-1]:
    
    Vl1,Vt1 = calcAveragedCurrents(t,dur_waves_best,dur_wl_best,dtime_waves,dtime_wl,waves,wl1)
    Vl.append(Vl1)
    Vt.append(Vt1)
    
reg = LinearRegression().fit(np.hstack([np.array(Vl)[~np.isnan(Vl)].reshape(-1,1),np.array(Vt)[~np.isnan(Vl)].reshape(-1,1)]),np.array(a)[~np.isnan(Vl)].reshape(-1,1))
yhat = reg.predict(np.hstack([np.array(Vl)[~np.isnan(Vl)].reshape(-1,1),np.array(Vt)[~np.isnan(Vl)].reshape(-1,1)]))
sst = np.sum((a[~np.isnan(Vl)]-np.mean(a[~np.isnan(Vl)]))**2)
ssr = np.sum((yhat-np.mean(a[~np.isnan(Vl)]))**2)
coef = (ssr/sst)

xb_x = [(i-datetime(2019,1,1)).total_seconds() for i in dtime_area]
xb = LinearRegression().fit(np.array(xb_x).reshape(-1,1),np.array(a).reshape(-1,1)).predict(np.array(xb_x).reshape(-1,1))
BSS = 1 - (np.sum((abs(np.array(area)[~np.isnan(Vl)].reshape(-1,1)-yhat)-np.array(2*np.std(area_e,axis=0)).reshape(-1,1)[~np.isnan(Vl)])**2)/np.sum(((np.array(area)[~np.isnan(Vl)].reshape(-1,1)-xb[~np.isnan(Vl)].reshape(-1,1))**2)))

# Visualize the model output compared to observations #
fig = plt.figure(figsize=(4,2))
ax = plt.axes([.16,.1,.8,.85])
ax1 = plt.axes([.26,.75,.1,.15])
ax2 = plt.axes([.43,.75,.1,.15])
ax3 = plt.axes([.8,.22,.15,.2])
h1 = ax.plot(dtime_area[1:-1],np.mean(area_e,axis=0)[1:-1],'ks-')
for i in range(0,len(shorelines3.keys())):
    ax.plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax.plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
    ax.plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax.plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
ax.set_xlim(dtime_area[1]-timedelta(days=1),dtime_area[-2]+timedelta(days=1))
ax.set_xticks(dtime_area[1:-1:5])
h2 = ax.plot(np.array(dtime_area)[~np.isnan(Vl)][1:-1],yhat[1:-1],'rs-')
ax.set_ylabel('Area ($m^2$)')
ax.set_ylim(5800,7850)
ax.set_xticks([datetime(2020,5,1),datetime(2020,5,10),datetime(2020,5,20),datetime(2020,5,30)])
ax1.plot(dtime_area[0],np.mean(area_e,axis=0)[0],'ks')
ax1.plot((dtime_area[0],dtime_area[0]),(np.mean(area_e,axis=0)[0],np.mean(area_e,axis=0)[0]+(2*np.std(area_e,axis=0)[0])),'k')
ax1.plot((dtime_area[0],dtime_area[0]),(np.mean(area_e,axis=0)[0],np.mean(area_e,axis=0)[0]-(2*np.std(area_e,axis=0)[0])),'k')
ax1.plot(dtime_area[0],yhat[0],'rs')
ax1.set_xticks([datetime(2020,3,5)])
ax1.set_ylim(ax.get_ylim())
ax1.set_yticks([6500,7500])
ax2.plot(dtime_area[-1],np.mean(area_e,axis=0)[-1],'ks')
ax2.plot((dtime_area[-1],dtime_area[-1]),(np.mean(area_e,axis=0)[-1],np.mean(area_e,axis=0)[-1]+(2*np.std(area_e,axis=0)[-1])),'k')
ax2.plot((dtime_area[-1],dtime_area[-1]),(np.mean(area_e,axis=0)[-1],np.mean(area_e,axis=0)[-1]-(2*np.std(area_e,axis=0)[-1])),'k')
ax2.plot(dtime_area[-1],yhat[-1],'rs')
ax2.set_xticks([datetime(2020,8,1)])
ax2.set_ylim(ax.get_ylim())
ax2.set_yticklabels([])
ax2.set_yticks([6500,7500])
ax3.plot(a[~np.isnan(Vl)],yhat,'k.',markersize=1.5)
ax3.plot((min(ax3.get_xlim()),max(ax3.get_xlim())),(min(ax3.get_xlim()),max(ax3.get_xlim())),'--',color='gray')
ax3.plot(a[~np.isnan(Vl)],yhat,'k.',markersize=3)
ax3.set_xticks([6500,7500])
ax3.set_yticks([6500,7500])
ax3.axis('equal')
ax3.set_xticklabels([6500,7500],rotation=-14)
ax3.set_yticklabels([6500,7500],rotation=50)
ax3.text(7400,6200,'Obs.',fontsize=6)
ax3.text(5800,7050,'Mod.',fontsize=6,rotation=90)
ax.legend([h1[0],h2[0]],['Observations','Model'],loc='lower left')
ax.text(0.95, 0.97, 'a', transform=ax.transAxes,fontsize=8, fontweight='bold', va='top',color='k')
ax3.text(0.85, 0.97, 'b', transform=ax3.transAxes,fontsize=8, fontweight='bold', va='top',color='k')





# Use the optimal model to predict area values #
waveobj_pred = utils.NDBCWaveRecord(41114,[2019,2020]) # Wave observstions from closest buoy #
waves_pred = waveobj_pred.download()  
dt = []
for ii in range(0,len(waves_pred)):
    i = waves_pred.iloc[ii]
    dt.append(datetime(int(i['yr']),int(i['mo']),int(i['day']),int(i['hr']),int(i['mm'])))
dtime_waves_pred = [i-timedelta(hours=4) for i in dt]   # convert from utc to edt #
i = waves_pred['MWD (degT)'] == 999
waves_pred['MWD (degT)'][i] = np.nan
i = waves_pred['MWD (degT)'] == 9999
waves_pred['MWD (degT)'][i] = np.nan
  
wlobj_pred = utils.NOAAWaterLevelRecord(8721604,'2019','2020') # Water level observations from other of two closest stations- Trident Pier #
wl_pred = wlobj_pred.get()
dtime_wl_pred= list(wl_pred['Time'])
dt = []
for i in dtime_wl_pred:
    dt.append(datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[14:16])))
dtime_wl_pred = dt


dtime_pred = []
for t in range(0,365):
    dtime_pred.append(datetime(2020,1,1)+timedelta(days=t))
Vl = []
Vt = []
for t in dtime_pred:
    Vl1,Vt1 = calcAveragedCurrents(t,dur_waves_best,dur_wl_best,dtime_waves_pred,dtime_wl_pred,waves_pred,wl_pred)
    Vl.append(Vl1)
    Vt.append(Vt1)   
dtime_pred = list(np.array(dtime_pred)[np.where(~np.isnan(Vl))[0]])
Vt = list(np.array(Vt)[np.where(~np.isnan(Vl))[0]])
Vl = list(np.array(Vl)[np.where(~np.isnan(Vl))[0]])
dtime_pred = list(np.array(dtime_pred)[np.where(~np.isnan(Vt))[0]])
Vl = list(np.array(Vl)[np.where(~np.isnan(Vt))[0]])
Vt = list(np.array(Vt)[np.where(~np.isnan(Vt))[0]])

area_pred = reg.predict(np.hstack([np.array(Vl).reshape(-1,1),np.array(Vt).reshape(-1,1)]))

dtt = [datetime(2020,1,1)+timedelta(days=int(i)) for i in np.arange(0,365)]
c = -1
area_pred_i = [None]*365
Vl_i = [None]*365
Vt_i = [None]*365
for i in dtt:
    c+=1
    try:
        p = int(np.where(np.array(dtime_pred)==i)[0])
    except TypeError:
        area_pred_i[c] = np.nan
        Vl_i[c] = np.nan
        Vt_i[c] = np.nan
    else:
        area_pred_i[c] = area_pred[p]
        Vl_i[c] = Vl[p]
        Vt_i[c] = Vt[p]

fig,ax = plt.subplots(3,figsize=(4.8,4))
h1 = ax[0].plot(dtime_area[0],np.mean(area_e,axis=0)[0],'ks')
ax[0].plot(dtime_area[1:-1],np.mean(area_e,axis=0)[1:-1],'ks-')
ax[0].plot(dtime_area[-1],np.mean(area_e,axis=0)[-1],'ks')
for i in range(0,len(shorelines3.keys())):
    ax[0].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i],dtime_area[i]),(np.mean(area_e,axis=0)[i],np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]+(2*np.std(area_e,axis=0)[i])),'k')
    ax[0].plot((dtime_area[i]-timedelta(hours=5),dtime_area[i]+timedelta(hours=5)),(np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i]),np.mean(area_e,axis=0)[i]-(2*np.std(area_e,axis=0)[i])),'k')
h2 = ax[0].plot(dtt,area_pred_i,'r')
ax[0].set_yticks([6000,6500,7000,7500])
ax[0].legend([h1[0],h2[0]],['Data','Model'],ncol=2,loc='lower right',handletextpad=0.2,columnspacing=1)
ax[0].set_ylabel('Area ($m^2$)')
ax[1].plot(dtt,Vl_i,'k')
ax[1].set_ylabel('$\hat V_l$ (m/s)')
ax[2].plot(dtt,Vt_i,'k')
ax[2].set_ylabel('$\hat V_t$ (m/s)')
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[0].text(0.01, 0.97, 'a', transform=ax[0].transAxes,fontsize=8, fontweight='bold', va='top',color='k')
ax[1].text(0.01, 0.97, 'b', transform=ax[1].transAxes,fontsize=8, fontweight='bold', va='top',color='k')
ax[2].text(0.01, 0.97, 'c', transform=ax[2].transAxes,fontsize=8, fontweight='bold', va='top',color='k')




















R2 = []
for dur_waves in np.arange(2,30*24):
        Hs = []
        w_cor = []  
        wl = []
        dur_wl = 12
        for w,t,n in zip(area[1:-2],dtime_area[1:-2],zv3[1:-2]):
            dt_start_waves = t-timedelta(hours=int(dur_waves))
            dt_start_wl = t-timedelta(hours=int(dur_wl))
            dt_end = t
    
            # Waves #
            i = 0
            val = dtime_waves[0]
            while val<dt_start_waves:
                i+=1
                val = dtime_waves[i]
            i_start_waves = i
            
            i = 0
            val = dtime_waves[0]
            while val<dt_end:
                i+=1
                val = dtime_waves[i]
            i_end_waves = i-1
        
            
            Hs.append(np.nanmean(np.array(waves.iloc[i_start_waves:i_end_waves]['wvht (m)'])))
            ###
            
            # Water level #
            i = 0
            val = dtime_wl[0]
            while val<dt_start_wl:
                i+=1
                val = dtime_wl[i]
            i_start_wl = i
            
            i = 0
            val = dtime_wl[0]
            while val<dt_end:
                i+=1
                val = dtime_wl[i]
            i_end_wl = i-1        
        
            wl.append(max(np.array(wl1.iloc[i_start_wl:i_end_wl]['wl_obs']))-min(np.array(wl1.iloc[i_start_wl:i_end_wl]['wl_obs'])))
            ###
        
        
            w_cor.append(w)
        try:
            reg = LinearRegression().fit(np.hstack([np.array(Hs).reshape(-1,1),np.array(wl).reshape(-1,1)]),np.array(w_cor).reshape(-1,1))
        except:
            R2.append(np.nan)
        else:
            yhat = reg.predict(np.hstack([np.array(Hs).reshape(-1,1),np.array(wl).reshape(-1,1)]))
            sst = np.sum((w_cor-np.mean(w_cor))**2)
            ssr = np.sum((yhat-np.mean(w_cor))**2)
            coef = (ssr/sst)
            R2.append(coef)

R2_max = max(R2)
R2_max_hours = np.arange(2,30*24)[np.where(R2==max(R2))[0]]


            
## Use the best model to predice the observable area until August 1 #
#t_start = datetime(2020,5,31,15,0)
#tt = []
#tt.append(t_start) 
#t_now = t_start
#while t_now<=dtime_area[-1]:
#    t_now = t_now+timedelta(days=1)
#    tt.append(t_now)
#
#a_pred = []    
#for t in tt:
#    dt_start = t-timedelta(hours=4)
#    dt_end = t
#
#    # Waves #
#    i = 0
#    val = dtime_waves[0]
#    while val<dt_start:
#        i+=1
#        val = dtime_waves[i]
#    i_start = i
#    
#    i = 0
#    val = dtime_waves[0]
#    while val<dt_end:
#        i+=1
#        val = dtime_waves[i]
#    i_end = i-1
#    
#    Hs = np.mean(np.array(waves.iloc[i_start:i_end]['wvht (m)']))
#    ###
#    
#    # Water level #
#    i = 0
#    val = dtime_wl[0]
#    while val<dt_start:
#        i+=1
#        val = dtime_wl[i]
#    i_start = i
#    
#    i = 0
#    val = dtime_wl[0]
#    while val<dt_end:
#        i+=1
#        val = dtime_wl[i]
#    i_end = i-1        
#    
#    wl = max(np.array(wl1.iloc[i_start:i_end]['wl_obs']))-min(np.array(wl1.iloc[i_start:i_end]['wl_obs']))
#    
#    # Make the prediction #
#    a_pred.append(float(reg.predict(np.array([Hs,wl]).reshape(1,-1))))
#
#plt.plot(dtime_area,area,'bs-')
#plt.plot(tt,a_pred,'r')

        