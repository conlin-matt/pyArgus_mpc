"""
Top-levl script file to examine accuracy of remote trabsformation at Jupiter Inlet surfcam

*NOTE*: To use this code, you'll need to make your current working directory
        the directory in which this script lives.
        
Matthew P. Conlin
University of Florida
01/2022
"""

# Standard library imports #

# Third-party imports #
from matplotlib import colorbar, colors, image as mpimg, patches, pyplot as plt
import numpy as np
import pickle
from scipy.io import loadmat

# Project imports #
from pyArgus_mpc import computation as comp


#=============================================================================#
# Establish things we need later #
#=============================================================================#
dataDirec = '/Users/matthewconlin/Documents/Research/WebCAT/pyArgus_mpc_project/pyArgus_mpc/SurfcamArgus/GeophysicalAnalyses/JupiterInletFL/data'# os.getcwd()+'/data' # MAKE SURE YOUR CURRENT WORKING DIRECTORY IS WHERE THIS SCRIPT LIVES #

rectif_xmin = -200
rectif_xmax = 200
rectif_dx = .2
rectif_ymin = 250
rectif_ymax = 900
rectif_dy = .2

#=============================================================================#
# 1. Calibrate the camera using SurfRCaT and make figures
    
#=============================================================================#
# Refer to the SurfRCaT github repo and SoftwareX publication for details. This will
# result in a calibVals file which contains the solved-for calibration parameters.
# I will load this in below #
calibVals = np.loadtxt(dataDirec+'/calibVals.txt') # Get these by using SurfRCaT #

# Make a figure of the remote-GCPs used to complete the SurfRCaT calibration #
im = mpimg.imread(dataDirec+'/snap.png')
f = open(dataDirec+'/gcps_im.pkl','rb'); gcps_im = pickle.load(f)

fig = plt.figure(figsize=(4.5,3))
ax1 = plt.axes([.05,.58,.9,.37])
ax2 = plt.axes([.02,.26,.3,.25])
ax3 = plt.axes([.34,.26,.3,.25])
ax4 = plt.axes([.66,.26,.3,.25])
ax5 = plt.axes([.02,.05,.3,.3])
ax6 = plt.axes([.34,.05,.3,.3])
ax7 = plt.axes([.66,.05,.3,.3])

ax1.imshow(im)
ax1.set_xticks([])
ax1.set_yticks([])
for p in gcps_im:
    ax1.plot(p[0][0],p[0][1],'r+',markersize=6)
rect1 = patches.Rectangle((230,500),340,-200,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
rect2 = patches.Rectangle((1350,550),200,-100,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
rect3 = patches.Rectangle((1450,650),200,-100,edgecolor='k',facecolor='none',linewidth=1,linestyle='--')
con1 = patches.ConnectionPatch(xyA=(230,500),xyB=(400,350),coordsA="data",coordsB="data",axesA=ax1,axesB=ax2,color='k')
con2 = patches.ConnectionPatch(xyA=(1350,550),xyB=(1450,450),coordsA="data",coordsB="data",axesA=ax1,axesB=ax3,color='k')
con3 = patches.ConnectionPatch(xyA=(1650,650),xyB=(1550,550),coordsA="data",coordsB="data",axesA=ax1,axesB=ax4,color='k')
con1_1 = patches.ConnectionPatch(xyA=(400,480),xyB=(609,400),coordsA="data",coordsB="data",axesA=ax2,axesB=ax5,color='k')
con2_1 = patches.ConnectionPatch(xyA=(1450,550),xyB=(609,400),coordsA="data",coordsB="data",axesA=ax3,axesB=ax6,color='k')
con3_1 = patches.ConnectionPatch(xyA=(1550,650),xyB=(609,400),coordsA="data",coordsB="data",axesA=ax4,axesB=ax7,color='k')

ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
ax1.add_artist(con1)
ax1.add_artist(con2)
ax1.add_artist(con3)
#ax1.text(-120,100,'a',fontweight='bold',fontsize=8)

ax2.imshow(im)
ax2.set_xticks([])
ax2.set_yticks([])
for p in gcps_im:
    ax2.plot(p[0][0],p[0][1],'r+',markersize=10)
ax2.set_xlim(280,520)
ax2.set_ylim(480,350)
ax2.add_artist(con1_1)

ax3.imshow(im)
ax3.set_xticks([])
ax3.set_yticks([])
for p in gcps_im:
    ax3.plot(p[0][0],p[0][1],'r+',markersize=10)
ax3.set_xlim(1350,1550)
ax3.set_ylim(550,450)
ax3.add_artist(con2_1)

ax4.imshow(im)
ax4.set_xticks([])
ax4.set_yticks([])
for p in gcps_im:
    ax4.plot(p[0][0],p[0][1],'r+',markersize=10)
ax4.set_xlim(1450,1650)
ax4.set_ylim(650,550)
ax4.add_artist(con3_1)

ax5.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/Jupiter/Results_SurfRCaT/GCPs_lidar_ex_1.png'))
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylim(500,0)
ax5.plot(614.59,171.591249,'k+')
ax5.plot(408.0375,344.648749,'k+')
ax5.plot(876.9675,394.891249,'k+')

ax6.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/Jupiter/Results_SurfRCaT/GCPs_lidar_ex_2.png'))
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_ylim(700,0)
ax6.plot(732.736,414.351,'k+')

ax7.imshow(mpimg.imread('/Users/matthewconlin/Documents/Research/WebCAT/Applications/Jupiter/Results_SurfRCaT/GCPs_lidar_ex_3.png'))
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_ylim(700,0)
ax7.plot(517.149,466.725,'k+')




#=============================================================================#
# 2. Calculate and visualize the accuracy of the calibration by calculating checkpoint reprojection residuals

#=============================================================================#
# Calculate the residuals #
gcpFile = dataDirec+'/checkpoints_world.txt'
gcpxy = loadmat(dataDirec+'/checkpoints_im.mat')['UV']
checks = np.arange(1,39)
camLoc = (592268.60,2979958.33)
difs,resids,rmsResid,gcpXYreproj = comp.calcCheckPointResid(calibVals,gcpxy,gcpFile,checks,camLoc)

# Make a figure showing the residuals on a rectified image #
im = mpimg.imread(dataDirec+'/snap.png')
im_rectif,extents = comp.RectifyImage(calibVals,im,[rectif_xmin,rectif_xmax,rectif_dx,rectif_ymin,rectif_ymax,rectif_dy,0])

scale_r = np.linspace(1,.5,11)
scale_g = np.linspace(1,0,11)
scale_b = np.linspace(1,0,11)
carr = np.transpose(np.vstack([scale_r,scale_g,scale_b]))
clist = [tuple(i) for i in carr]
cm = colors.ListedColormap(clist)
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2.74,3.5))
ax1 = plt.axes([.2,.1,.6,.85])
cbax = plt.axes([.78,.1,.04,.85]) 
ax2 = plt.axes([.246,.65,.505,.30])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = plt.axes([.34,.72,.06,.2])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([])
ax4 = plt.axes([.53,.72,.195,.2])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax1.imshow(im_rectif,extent=extents)
for i in range(0,len(gcpXYreproj)):
    r = np.floor(resids[i])
    if r>10:
        ci=10
    else:
        ci=int(r)
    ax1.plot(gcpXYreproj[i][1]-592268.6,gcpXYreproj[i][2]-2979958.33,'o',markeredgecolor='k',markerfacecolor=clist[ci],markersize=5)
cb1=colorbar.ColorbarBase(cbax,cmap=cm,boundaries=[0,1,2,3,4,5,6,7,8,9,10,11],ticks=[0,1,2,3,4,5,6,7,8,9,10],spacing='proportional',orientation='vertical',label='Reprojection residual (m)')
ax1.set_xlabel('Relative Easting (m)',fontsize=8)
ax1.set_ylabel('Relative Northing (m)',fontsize=8)
ax1.set_xticks([-200,-100,0,100])
ax1.set_xlim(-200,100)
ax3.boxplot(resids,sym='kx')
ax3.set_xticks([])
ax3.set_ylim(0,20)
ax3.set_yticks([0,5,10,15,20])
ax3.set_xlim(0.85,1.15)
ax3.set_xlabel('Residuals',fontsize=6)
ax4.plot(abs(difs[:,0]),abs(difs[:,1]),'.',markersize=2)
ax4.plot(np.linspace(0,20,20),np.linspace(0,20,20),'k-')
ax4.axis('equal')
ax4.set_xticks([0,20])
ax4.set_yticks([0,20])
ax4.set_xlabel('E-W',labelpad=-12,fontsize=6)
ax4.set_ylabel('N-S',labelpad=-17,fontsize=6)
#ax1.text(0.02, 0.05, 'b', transform=ax1.transAxes,fontsize=8, fontweight='bold', va='top',color='w')



