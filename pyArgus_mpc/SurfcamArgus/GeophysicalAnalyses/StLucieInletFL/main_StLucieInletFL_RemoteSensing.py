"""
Top-levl script file to perform analyses at the St. Lucie Inlet surfcam

The videos for the shoreline change analysis are in a figshare repository and can
be found and downloaded at https://doi.org/10.6084/m9.figshare.c.5816801

*NOTE*: To use this code, you'll need to make your current working directory
        the directory in which this script lives. You will also need to set
        the vidDirec variable on Line 36 to be the directory where you saved
        the videos from the repo.
        
Matthew P. Conlin
University of Florida
01/2022
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
from scipy.io import loadmat

# Project imports #
from pyArgus_mpc import computation as comp, shorelineMapping as sl, utils
from pyArgus_mpc.SurfcamArgus import analysisTools as sca
from pyArgus_mpc.SurfcamArgus.GeophysicalAnalyses import vidPrep


#=============================================================================#
# Establish things we need later #
#=============================================================================#
vidDirec = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/StLucie/RawVideoData/2020-05' # CHANGE THIS TO THE DIRECTORY WHERE YOU SAVED THE VIDEOS #
dataDirec = '/Users/matthewconlin/Documents/Research/WebCAT/pyArgus_mpc_project/pyArgus_mpc/SurfcamArgus/GeophysicalAnalyses/StLucieInletFL/data'# os.getcwd()+'/data' # MAKE SURE YOUR CURRENT WORKING DIRECTORY IS WHERE THIS SCRIPT LIVES #

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
f = open(dataDirec+'/calibVals.pkl','rb') # Get these from SurfCaT #
calibVals = pickle.load(f)

# Make a figure of the remote-GCPs used to complete the SurfRCaT calibration #
im = mpimg.imread(dataDirec+'/snap.png')
f = open(dataDirec+'/gcps_im.pkl','rb'); gcps_im = pickle.load(f)

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
ax3.imshow(mpimg.imread(dataDirec+'/gcps_world_ims/GCPs_lidar_ex_1.png'))
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
ax5.imshow(mpimg.imread(dataDirec+'/gcps_world_ims/GCPs_lidar_ex_2.png'))
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
ax7.imshow(mpimg.imread(dataDirec+'/gcps_world_ims/GCPs_lidar_ex_3.png'))
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
ax9.imshow(mpimg.imread(dataDirec+'/gcps_world_ims/GCPs_lidar_ex_4.png'))
ax9.set_xticks([])
ax9.set_yticks([])
ax9.plot(623.6395073180788,290.35301999587693,'k+')



#=============================================================================#
# 2. Calculate and visualize the accuracy of the calibration by calculating checkpoint reprojection residuals

#=============================================================================#
gcpFile = dataDirec+'/checkpoints_world.txt'
gcpxy = loadmat(dataDirec+'/checkpoints_im.mat')['UV']
checks = np.arange(1,32)
camLoc = (583381.79,3005482.72)
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
# Establish transects #
def createTransect(endPts):
    import skimage.transform as transform
    m = 90-np.degrees(np.tan((endPts[1][0]-endPts[0][0])/(endPts[1][1]-endPts[0][1])))
    if np.sign(endPts[1][1]-endPts[0][1])==-1:
        phi = (m+180)*np.pi/180
    else:
        phi = (m)*np.pi/180
    length = np.sqrt( (endPts[1][0]-endPts[0][0])**2 + (endPts[1][1]-endPts[0][1])**2 )
    x = np.linspace(0,length,length+1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y
    tf = transform.EuclideanTransform(rotation=phi,translation=(endPts[0][0],endPts[0][1]))
    transect = tf(coords)
    return transect
#fig,ax = plt.subplots(1)
#for key in shorelines3.keys():
#    ax.plot(shorelines3[key][:,0],shorelines3[key][:,1])
#ax.axis('equal')
#plt.ginput(2)
t_east = [(260.2195564516152, -111.60504225155698),
 (264.4550000000023, -147.37100999349298)]
t_west = [(188.2170161290335, -103.60475999349235),
 (190.57004032258192, -138.42951805800897)]
t_back = [(167.44677419354932, -102.0216740094109),
 (162.27500000000086, -75.22248046102342)]
transect_east = createTransect(t_east)
transect_west = createTransect(t_west)
transect_back = createTransect(t_back)

#fig,ax = plt.subplots(1)
#ax.plot(shorelines3['202005011116'][:,0],shorelines3['202005011116'][:,1])
#ax.axis('equal')
#ax.plot(transect_east[:,0],transect_east[:,1],'r')
#ax.plot(transect_west[:,0],transect_west[:,1],'r')
#ax.plot(t_east[0][0],t_east[0][1],'g*')
#ax.plot(t_east[1][0],t_east[1][1],'g*')
#ax.plot(t_west[0][0],t_west[0][1],'g*')
#ax.plot(t_west[1][0],t_west[1][1],'g*')

# Map the spit shorleines #
direc = vidDirec
zv3 = []
f = open(dataDirec+'/CCD_extents.pkl','rb');h = pickle.load(f)
shorelines3 = {}
ii = -1
for vid in sorted([i for i in os.listdir(direc) if '.ts' in i or '.avi' in i]):
    
    ii+=1
    date = vid.split('_')[1].split('.')[0]
    
    # Reduce the video #
    if not os.path.exists(direc+'/StLucie_'+date+'_reduced.avi'):
        vidPrep.ReduceVid(direc+'/StLucie_'+date+'.ts') # Reduce the vid to 1 frame per sec
     
    # Make the timex #    
    if not os.path.exists(direc+'/StLucie_'+date+'_timex.png'):
        timex = sca.CreateImageProduct(direc+'/StLucie_'+date+'_reduced.avi',1)
        
        timex_towrite = np.stack([timex[:,:,2]*255,timex[:,:,1]*255,timex[:,:,0]*255],axis=2)
        cv2.imwrite(direc+'/StLucie_'+date+'_timex.png',timex_towrite) 
    else:
        timex = mpimg.imread(direc+'/StLucie_'+date+'_timex.png')
        
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

# Make a figure of shoreline maps #
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(6.5,7))
ax = fig.subplots(9,4)  
it = -1
for timex,shoreline,axisRow,axisCol in zip(sorted([i for i in os.listdir(direc) if 'timex' in i]),shorelines3,[0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8],
                                           [0,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,3]):
    
    it += 1
    timex1 = timex
    timex = mpimg.imread(direc+'/'+timex)
   
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
            ax[axisRow][axisCol].plot(transect_east[:,0],transect_east[:,1],'r')
            ax[axisRow][axisCol].plot(transect_west[:,0],transect_west[:,1],'r')
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



############################################################
# Intersect shorelines with transects
###########################################################
intersections_unc = []
for itt in range(0,2):
    iRow = -1
    intersections = np.zeros([len(shorelines3),2])
    it = -1
    for key in shorelines3.keys():
            iRow+=1
            it+=1
            f = shorelines3[key]
            x = f[:,0]
            y = f[:,1]
            
            # Correct for water level by shifting the outline everywhere in the cross shore to a reference tidal value of 0 m #
            z_tide = zv3[it]
            z_ref = -0.29
            m = 0.1
            d = -(z_ref-z_tide)/m
            xyz_new = np.empty([0,2])
            for ii in range(1,len(x)-1):
                theta = math.atan2(y[ii+1]-y[ii-1],x[ii+1]-x[ii-1])
    #            plt.plot(x[ii-1:ii+2],y[ii-1:ii+2],'b.')
    #            plt.plot((x[ii-1],x[ii+1]),(y[ii-1],y[ii+1]),'r')
                theta_norm = theta+(math.pi/2)
                x_n = d*math.cos(theta_norm)+x[ii]
                y_n = d*math.sin(theta_norm)+y[ii]         
    #            plt.plot(x_n,y_n,'g.')
                xyz_new = np.vstack([xyz_new,(x_n,y_n)])
            
            del x,y
            x = xyz_new[:,0]
            y = xyz_new[:,1]
            
            # Calculate the intersection with both transects #
            sl = np.transpose(np.vstack([x,y]))
            ts = [transect_east,transect_west]
            iCol = -1
            for t in ts:
                iCol+=1
                # Rotation matrix #
                X0 = t[0,0]
                Y0 = t[0,1]
                temp = np.array(t[-1,:]) - np.array(t[0,:])
                phi = np.arctan2(temp[1],temp[0])
                Mrot = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
                # point to line distance between shoreline points and the transect #
                p1 = np.array([X0,Y0])
                p2 = t[-1,:]
                d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
                # distance between shoreline points and transect origin #
                d_origin = np.array([np.linalg.norm(sl[k,:]-p1) for k in range(len(sl))])
                # find shoreline points that are close to the transect and its origin #
                idx_dist = np.logical_and(d_line<=10,d_origin<=1000)
                # find the shoreline points that are in the direction of the transect (??) #
                temp_sl = sl - np.array(t[0,:])
                phi_sl = np.array([np.arctan2(temp_sl[k,1],temp_sl[k,0]) for k in range(len(temp_sl))])
                diff_angle = (phi-phi_sl)
                idx_angle = np.abs(diff_angle)<np.pi/2
                # combine the points that are close in distance and orientation #
                idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]
                # change of base to shore-normal coordinate system #
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],[Y0]]),(1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot,xy_close)
                # compute the median of the intersections along the transect #
                intersection = np.nanmedian(xy_rot[0,:])
                intersections[iRow,iCol] = intersection
                                
    intersections_unc.append(intersections)

intersections_all_t1 = np.empty([len(intersections),0])   
intersections_all_t2 = np.empty([len(intersections),0])   
for i in range(0,len(intersections_unc)):
    intersections_all_t1 = np.hstack([intersections_all_t1,intersections_unc[i][:,0].reshape(-1,1)])
    intersections_all_t2 = np.hstack([intersections_all_t2,intersections_unc[i][:,1].reshape(-1,1)])
intersections_t1_mean = np.mean(intersections_all_t1,axis=1)
intersections_t1_std = np.std(intersections_all_t1,axis=1)
intersections_t2_mean = np.mean(intersections_all_t2,axis=1)
intersections_t2_std = np.std(intersections_all_t2,axis=1)          
time = [datetime.strptime(i,'%Y%m%d%H%M') for i in list(shorelines3.keys())]   



# Make the shoreline change figure #
fig,ax = plt.subplots(2,1,sharex=True,figsize=(6.5,3))
ax[0].bar(x=time[1:-8:6],height=np.diff(intersections_t1_mean[1:-2:6]),width=[7,7,7,7,],align='edge',edgecolor='k')
ax[0].bar(x=time[0:-1],height=np.diff(intersections_t1_mean),width=np.diff(time),align='edge',edgecolor='k')
ax[0].plot(ax[0].get_xlim(),(6.2,6.2),'k--')
ax[0].plot(ax[0].get_xlim(),(-6.2,-6.2),'k--')
ax[0].set_xticks(time[1:len(time):6])
ax[0].set_ylabel('$\Delta x$ (m)')
ax[0].set_ylim([-15,10])
ax[0].text(datetime(2020,5,15),-14,'a (East Transect)',fontweight='bold',ha='center')

ax[1].bar(x=time[1:-8:6],height=np.diff(intersections_t2_mean[1:-2:6]),width=[7,7,7,7,],align='edge',edgecolor='k')
ax[1].bar(x=time[0:-1],height=np.diff(intersections_t2_mean),width=np.diff(time),align='edge',edgecolor='k')
ax[1].plot(ax[1].get_xlim(),(6.1,6.1),'k--')
ax[1].plot(ax[1].get_xlim(),(-6.1,-6.1),'k--')
ax[1].set_xticks(time[1:len(time):6])
ax[1].set_ylabel('$\Delta x$ (m)')
ax[1].set_ylim([-40,10])
ax[1].text(datetime(2020,5,15),-37,'b (West Transect)',fontweight='bold',ha='center')







