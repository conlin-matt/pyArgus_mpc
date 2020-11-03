#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:23:11 2020

@author: matthewconlin
"""

# Standard library imports #
import math

# Third-party imports #
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import interp1d



class NOAAWaterLevelRecord():
    
    '''
    Class to interact with NOAA water level records. Class is initialized with desired station and record time.
    
    Inputs:
        station: (int) NDBC station from which to extract data
        bdate: (int or str): the date for the start of the record in the form yyyymmddHHMMSS
        ebdate: (int or str): the date for the end of the record in the form yyyymmddHHMMSS

    
    Methods:
        get(): Download water level record of initialized duration from initialized station. 
        atTime(): Get the water level at a specific time. Must have used get() first.
        
    '''
    
    def __init__(self,station,bdate,edate):
        self.station=station
        self.bdate = bdate
        self.edate = edate
        
    
    def get(self):
       
        product = 'hourly_height&application=UF'
        product2 = 'predictions&application=UF'
        
        api = 'https://tidesandcurrents.noaa.gov/api/datagetter?product='+product+'&begin_date='
        url = api+str(self.bdate)+'&end_date='+str(self.edate)+'&datum=NAVD&station='+str(self.station)+'&time_zone=lst_ldt&units=metric&format=csv'
        api2 = 'https://tidesandcurrents.noaa.gov/api/datagetter?product='+product2+'&begin_date='
        url2 = api2+str(self.bdate)+'&end_date='+str(self.edate)+'&datum=NAVD&station='+str(self.station)+'&time_zone=lst_ldt&units=metric&interval=h&format=csv'
        
        
        dat_obs = pd.read_csv(url)
        dat_pred = pd.read_csv(url2)
        
        wl = pd.DataFrame({'Time':dat_obs['Date Time'],'wl_obs':dat_obs[' Water Level'],'wl_pred':dat_pred[' Prediction']})
        
        self.wl = wl
        return wl
    
    def atTime(self,wl,date):
        d = pd.to_datetime(self.wl['Time'])
        # Turn the date times into numeric values (seconds since the first observation)
        time_int = []
        for i in d:
            td = i-d[0]
            time_int.append(td.total_seconds())
            
        # Get the numeric value for the desired date #
        td = pd.to_datetime(date)-d[0]
        timeWant = td.total_seconds()
        
        # Interpolate wl at that time #
        f = interp1d(time_int,self.wl['wl_obs'])
        wli = float(f(timeWant))
        
        return wli




class NDBCWaveRecord():
    
    '''
    Class to auto-download data from NDBC station. Class can return downloaded data and
    can extract a parameter value at a user-input time.
    
    Inputs:
        station: (int) NDBC station from which to extract data
        years: (int or list of ints) Year(s) for which to extract data
        
    Methods:
        download(): Downlod the record.
        atTime(): Get the value of a specified parameter at a certain time.
    '''
    
    def __init__(self,station,years):
        self.station = station
        self.years = years
        
    def download(self):
        '''
        Returns a dataframe containing the data.
        '''
        
        if type(self.years) is int:
            self.years = [self.years]
        
        allDat = None  
        for yr in self.years:
            
            if yr != datetime.datetime.now().year:
                url = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='+str(self.station)+'h'+str(yr)+'.txt.gz&dir=data/historical/stdmet/'
                dat = pd.read_csv(url,sep=' ',delimiter=' ',header=2,index_col=False,usecols=[0,1,2,3,4,9,11,13,15],
                                  names=['yr','mo','day','hr','mm','wvht (m)','DPD (sec)','APD (sec)','MWD (degT)'])
            else:
                for i in range(1,datetime.datetime.now().month):
                    try:
                        datetime_object = datetime.datetime.strptime(str(i), "%m")
                        month_name = datetime_object.strftime("%b")
                        
                        url = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='+str(self.station)+str(i)+str(yr)+'.txt.gz&dir=data/stdmet/'+month_name+'/'
               
                        dat1 = pd.read_csv(url,sep=' ',delimiter=' ',header=1,index_col=False,usecols=[0,1,2,3,4,8,9,10,11],skipinitialspace=True,
                                           names=['yr','mo','day','hr','mm','wvht (m)','DPD (sec)','APD (sec)','MWD (degT)'])
                        if i == 1:
                            dat = dat1
                        else:
                            dat = dat.append(dat1)
                        
                    except:
                        break
                
            if allDat is not None:
                allDat = allDat.append(dat)
            else:
                allDat = dat
         
        allDat.set_index(np.arange(0,len(allDat)),inplace=True)    
        return allDat
        
    def atTime(self,dat,date,param):
        
        if param==1:
            p = 'wvht (m)'
        elif param==2:
            p = 'DPD (sec)'
        elif param==3:
            p = 'APD (sec)'
        elif param==4:
            p = 'MWD (degT)'
        
        dtimes = []
        for i in range(0,len(dat)):
            if dat['mo'][i]<10:
                dummo = '0'
            else:
                dummo = ''
            if dat['day'][i]<10:
                dumday = '0'
            else:
                dumday = ''
            if dat['hr'][i]<10:
                dumhr = '0'
            else:
                dumhr = ''
            if dat['mm'][i]<10:
                dummm = '0'
            else:dummm = ''
            
            dtimes.append(str(dat['yr'][i])+dummo+str(dat['mo'][i])+dumday+str(dat['day'][i])+dumhr+str(dat['hr'][i])+dummm+str(dat['mm'][i]))
            
        d = pd.to_datetime(dtimes)
        # Turn the date times into numeric values (seconds since the first observation)
        time_int = []
        for i in d:
            td = i-d[0]
            time_int.append(td.total_seconds())
            
        # Get the numeric value for the desired date #
        td = pd.to_datetime(date)-d[0]
        timeWant = td.total_seconds()
        
        # Interpolate wl at that time #
        f = interp1d(time_int,dat[p])
        pi = float(f(timeWant))
        
        return pi
    
       
        
def newtRaph(T,h):
    
    '''
    Function to determine k from dispersion relation given period (T) and depth (h) using
    the Newton-Raphsun method.
    '''
    
    L_not = (9.81*(T**2))/(2*math.pi)
    k1 = (2*math.pi)/L_not

    def fk(k):
        return (((2*math.pi)/T)**2)-(9.81*k*math.tanh(k*h))
    def f_prime_k(k):
        return (-9.81*math.tanh(k*h))-(9.81*k*(1-(math.tanh(k*h)**2)))

    k2 = 100
    i = 0
    while abs((k2-k1))/k1 > 0.01:
          i+=1
          if i!=1:
              k1=k2
          k2 = k1-(fk(k1)/f_prime_k(k1))

    return k2


    
def ShorelineElevation(wl,Hs,Hrms,d,T,A,beta,kk,gamma):        
    
    '''
    Determine the elevation of the shoreline using the setup model of Battjes and Janssen (1978), astronomical tide, and a 
    swash parameterization described in Aarninkhof et al. (2003). The setup model is run over an exponential approx.
    of the actual beach profile.
    
    Inputs:
        wl (float): measured water level at the time of shorline observation
        Hs (float): offshore significant wave height at the time of shoreline observation
        Hrms (float): offshore RMS wave height at the time of shoreline obsevation
        d  (float): The distance offshore to which the exponential beach profile extends
        T (float): Measured wave period at the time of shoreline observation
        A (float): The A parameter in the y=AX^(2/3) beach profile approximation, controlled by grain size.
        beta (float): Intertidal beach slope
        kk (float): constant used to scale the swash component of the water level. See Plant et al. (2007, JCR) for more info
        gamma (float): The depth-limited breaking criterion (H/h)
        
    Returns:
        setup (float): the computed wave setup elevation
        swash (float): the computed swash elevation
        z_sl (float): the computed elevation of the shoreline (water level+setup+swash)
    '''

    def BattjesSetupModel(Hrms,d,T,A,gamma):
        
        def calcQ(Q,c):
            return Q+(c*math.log(Q))-1
 
        # Establish the exponential beach profile (y=Ax^(2/3)) #
        dx = 1
        maxX = (d/A)**(3/2)
        x = np.arange(0,maxX,dx)
        y = A*np.power(x,(2/3))
        
        # Establish needed constants #
        rho = 1025
        g = 9.81
        
        # Initialize at off-shore most point (assume setup=0)
        f = (2*math.pi/T)
        k = newtRaph(T,y[len(y)-1]) # Calculate initial k based on single f that does not change #
        Hm = 0.88*(k**(-1))*math.tanh(gamma*k*y[len(y)-1]/0.88)
        c = -(Hrms/Hm)**2
        try:
            Qb = float(optimize.root(calcQ,.001,args=(c))['x'])
        except ValueError:
            Qb = 0

        e = .25*f*Qb*rho*g*(Hm**2)
        P = ((1/8)*rho*g*(Hrms**2))*((2*math.pi*f/k)*(.5+(k*y[len(y)-1]/math.sinh(2*k*y[len(y)-1]))))
        Sxx = (.5+(2*k*y[len(y)-1]/math.sinh(k*y[len(y)-1])))*((1/8)*rho*g*(Hrms**2))
        s = 0
        
        # Iterativly calculate the model onshore #
        k_vec = [k]
        P_vec = [P]
        Hrms_vec = [Hrms]
        Hm_vec = [Hm]
        c_vec = [c]
        Qb_vec = [Qb]
        e_vec = [e]
        Sxx_vec = [Sxx]
        s_vec = [s]
        for ih in range(len(y)-2,0,-1):
            h = y[ih] #depth#
            k = newtRaph(T,h)#k at this depth #
        
            P_here = P-(dx*e)#Integrate the governing equation to get the wave energy density at this point #
        
            cg = ((2*math.pi*f/k)*(.5+(k*h/math.sinh(2*k*h)))) # New Hrms based on the new wave energy density value #
            Hrms_here = math.sqrt(P_here/(cg*(1/8)*rho*g))
        
            Hm_here = 0.88*(k**(-1))*math.tanh(gamma*k*h/0.88) #hmax at this depth #
        
            if Hrms_here>Hm: # Deal with discontinuity at shoreline, Battjes did this in the original paper #
                Hrms_here = Hm_here
        
            c_here = -(Hrms_here/Hm_here)**2 #new Qb value at this depth #
            try:
                Qb_here = float(optimize.root(calcQ,.001,args=(c_here))['x'])
            except ValueError:
                Qb_here = 0
        
            e_here = .25*f*Qb_here*rho*g*(Hm_here**2) # New epsilon based on the new Qb #
        
        
            Sxx_here = (.5+(2*k*h/math.sinh(k*h)))*((1/8)*rho*g*(Hrms_here**2)) # Sxx here and change in Sxx since the last position #
            dSxxdx = (Sxx_here-Sxx)/dx
        
            s_here = s+(dx*(-dSxxdx/(rho*g*(h+s)))) # new setup value #
        
            k_vec.append(k)
            P_vec.append(P_here)
            Hrms_vec.append(Hrms_here)
            Hm_vec.append(Hm_here)
            c_vec.append(c_here)
            Qb_vec.append(Qb_here)
            e_vec.append(e_here)
            Sxx_vec.append(Sxx_here)
            s_vec.append(s_here)
        
            P = P_here
            e = e_here
            Sxx = Sxx_here
            s = s_here
            
        return max(s_vec)
    
    def swashComponent(Hs,T,beta,kk):
        
        # Inputs #
        Lo = (9.81*(T**2))/(2*math.pi)
        
        # Calc Irrabarren number #
        e = beta/math.sqrt(Hs/Lo)
        
        # Calc the two components, Rig and Rss #
        Rig = 0.65*Hs*math.tanh(3.38*e)
        if e>0.275:
            Rss = (0.69*e*Hs) - (0.19*e)
        else:
            Rss = 0
        
        # Calc the final term #
        swash = math.sqrt(Rig**2 + Rss**2)
        swashTerm = kk*(swash/2)    

        return swashTerm                                  
        
        
    setup = BattjesSetupModel(Hrms,d,T,A,gamma)
    swash = swashComponent(Hs,T,beta,kk)
       
    z_sl = wl+setup+swash
    
    return setup,swash,z_sl



def zoom(ax,base_scale = 2.):
    
    '''
    Allow interactive zooming in an image with the scroll wheel.
    
    Inputs:
        ax: The axis in which to do the zooming
        
    Returns:
        zoom_fun: The inner function performing the zooming
        
    '''
        
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun
