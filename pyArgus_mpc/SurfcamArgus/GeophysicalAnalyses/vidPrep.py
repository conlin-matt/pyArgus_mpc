#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:58:13 2020

@author: matthewconlin
"""

# Standard library imports #
import os 

# Third-party imports #
import cv2
import numpy as np
import requests




def GrabGDVid(id, destination):
    
    '''
    Grab video off of Google Drive and save it to destination given the file's shareable link.
    '''
    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token=value
        else:
            token=None
    if not token:
        token=None

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)


    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk) 




def ReduceVid(vid):
    
    '''
    Reduce a video to a frame rate of 1 Hz. Video plays at 2 Hz.
    '''
    
    # Get video properties #
    cap = cv2.VideoCapture(vid)
    numFrames = int(cap.get(7))
    fps = cap.get(5)
    
    # Reduce the video to one frame per second and save it as a new video. This reduces the number of frames in the video by an order of magnitude or more. #
    cap.set(1,1) 
    test,im = cap.read()
    
    name = vid.rsplit('.',1)[0]+'_reduced.avi'
    out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'MJPG'),1,(np.shape(im)[1],np.shape(im)[0])) # Video writer object #
    
    # Write 1 frame per second (the middle frame of that second) #
    totalFrames = 0
#    for i in range(0,int(round(numFrames/fps))):
    for i in range(0,600): # cut off at 10 min #
    
        # Pull the frame from midway through this second #
        frame = totalFrames+(fps/2)
        # Read that frame and cut the resolution in half #
        cap.set(1,frame) 
        test,im = cap.read()
        # Put the frame in the video #
        out.write(im)
        # Update frame count #
        totalFrames = totalFrames+fps
    cap.release()
    out.release()



def prep_JupiterInletFL():
    
    '''
    Function calls the GrabGDVid and ReduceVid functions in this module to pepare the videos for Jupiter Inlet, FL.
    Also formats video links in a dictionary.
    '''
    
    #=============================================================================#
    # Make dictionary containing date, time, and link for every video #
    #=============================================================================#
    links = {'2020-03-07':np.vstack([np.hstack([800,'https://drive.google.com/file/d/1AIDlvXQs4cZYyiW-lXdvnbwi6g_u3KCe/view?usp=sharing']),
                                     np.hstack([900,'https://drive.google.com/file/d/1hzTApXS9UxzjH4EO46hXxu9_l1d3C8aE/view?usp=sharing']),
                                     np.hstack([1000,'https://drive.google.com/file/d/1Yr7IVP6uPAdn0if4vC_R7KQDRt19nsIR/view?usp=sharing']),
                                     np.hstack([1100,'https://drive.google.com/file/d/1XIyGHezvbuoYKVnpj4zgvDv3j5nhnS5V/view?usp=sharing']),
                                     np.hstack([1200,'https://drive.google.com/file/d/1mwIh2dbQ9i0kx_yhgtweKU1KUs3wbbvt/view?usp=sharing']),
                                     np.hstack([1300,'https://drive.google.com/file/d/1Y7szCK2G1YSKze-p0qp8sSIJNqKrPojS/view?usp=sharing']),
                                     np.hstack([1400,'https://drive.google.com/file/d/1_GxXnh8mAs5ubWZSUAYybblD3MCuTAt3/view?usp=sharing']),
                                     np.hstack([1500,'https://drive.google.com/file/d/1LW-nzsi753tkga5fBOaEEzs4fVxLlZaO/view?usp=sharing']),
                                     np.hstack([1600,'https://drive.google.com/file/d/1kQWIQ94GIqeVP80azoiYudY_4YZURB7m/view?usp=sharing']),
                                     np.hstack([1800,'https://drive.google.com/file/d/1q3ceOGe6O8vNmtJhvVnCbNEyCm8sl0ea/view?usp=sharing'])
                                     ]),
            '2020-03-08':np.vstack([np.hstack([900,'https://drive.google.com/file/d/1wxRuZbNsHLF9oUjin3w0q_dY3dOSrgzl/view?usp=sharing']),
                                    np.hstack([1000,'https://drive.google.com/file/d/1OZcdi2tf1yJ5wEoKcPLM7eraOQF-GrSi/view?usp=sharing']),
                                    np.hstack([1100,'https://drive.google.com/file/d/10W4W0NS63LN59HHcbJe3syFovViMAzbq/view?usp=sharing']),
                                    np.hstack([1200,'https://drive.google.com/file/d/1o5TBLyz38IIhm_yZDquBZyPHYqpq7rU3/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.com/file/d/1LY_0hTis6w-nlE2O9QdgYuGIRP924YFK/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/1ZeKk3ezoX4zyDg-sXo6kyNX4kruqAF3q/view?usp=sharing']),
                                    np.hstack([1500,'https://drive.google.com/file/d/1Lp36uubZPR2zFC5thnlT8INaMYVDeiAa/view?usp=sharing']),
                                    np.hstack([1800,'https://drive.google.com/file/d/1rTrUiAedHY4p-jpEyKyNIWBySJFwGtg6/view?usp=sharing']),
                                    ]),
            '2020-03-09':np.vstack([np.hstack([1200,'https://drive.google.com/file/d/1lhP2LfPNXrCv-6X9pLuHKT6w_XmUFsBQ/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.com/file/d/1rTA8nzuGutfQ2-5Uli9hTebWHa1ab0eo/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/1wMHzxf9PUipBaXYYbtNxAqZnXg1LuqFQ/view?usp=sharing']),
                                    np.hstack([1500,'https://drive.google.com/file/d/1uf9KrFksj591XGIK7dXxnnZ6nnyTxvWF/view?usp=sharing']),
                                    np.hstack([1600,'https://drive.google.com/file/d/10pbnmqSjiPYuJzZw-AfTirbunvmtoDsE/view?usp=sharing']),
                                    np.hstack([1700,'https://drive.google.com/file/d/1BQlIXdJDNBbB6K1JzCQr9j-ThWeosVV4/view?usp=sharing'])
                                    ]),
            '2020-03-10':np.vstack([np.hstack([800,'https://drive.google.com/file/d/17Dv1Oc3njcSlT3TZLwam8kdB5Z682dzD/view?usp=sharing']),
                                    np.hstack([900,'https://drive.google.com/file/d/199cQ4EEPjjsdFS1acOl13-T4BrB5RDmF/view?usp=sharing']),
                                    np.hstack([1000,'https://drive.google.com/file/d/1CRk4KgSZFeORdg8ubazVbSKJYgXN_469/view?usp=sharing']),
                                    np.hstack([1100,'https://drive.google.com/file/d/1newT8V-DAKnK4_XKHRKMU7LlHNvYHQaE/view?usp=sharing']),
                                    np.hstack([1200,'https://drive.google.com/file/d/1zRs9lgDoCLaEYQF6AK6GJO0omffvJXMZ/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.com/file/d/1NjEARyXqQwoPvG1A1krGkDGcGOfF7ddB/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/1R6Gsirjc359mMvCU4DapAFvjS3x9xbru/view?usp=sharing']),
                                    np.hstack([1500,'https://drive.google.com/file/d/1f42FSqSvCEq-MzZ9S58TetzTyzn1mHp8/view?usp=sharing']),
                                    np.hstack([1600,'https://drive.google.com/file/d/1t8xZxERHWeHsiCFjMC9bfDwJp_JJf6Ym/view?usp=sharing']),
                                    np.hstack([1700,'https://drive.google.com/file/d/1fwrHopPfvi34AIdfFNgiV63FT2EdR0cn/view?usp=sharing']),
                                    np.hstack([1800,'https://drive.google.com/file/d/1pD_gJ7GOdSmh1yJfkCFCx0x2FeMtDJhJ/view?usp=sharing'])
                                    ]),
            '2020-03-12':np.vstack([np.hstack([1100,'https://drive.google.com/file/d/11TVa_GS3IqsnEm2h7mWGSlGKASxN7sPR/view?usp=sharing']),
                                    np.hstack([1200,'https://drive.google.com/file/d/1CD0SMIzncGDvKOVAvcDnuUAhgLKBZqV3/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.222/file/d/15BsGOymWzuNTmr0_2Hz7KPK8k00lPtLL/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/1TNaXzHMfePle17XnAePUWVNOHo7_8zhB/view?usp=sharing']),
                                    np.hstack([1500,'https://drive.google.com/file/d/1YGN3OZk8jY6vO-zKc9QfGlPF0jXYTFxG/view?usp=sharing']),
                                    np.hstack([1600,'https://drive.google.com/file/d/1VA2CN_2pTvNJATEyj2qQlPERlZbHX59Z/view?usp=sharing']),
                                    np.hstack([1700,'https://drive.google.com/file/d/1cP4a1YD3AJ04asqPrJQQhrxEHSDdofW_/view?usp=sharing']),
                                    np.hstack([1800,'https://drive.google.com/file/d/1uIJGrgAMxscFsgMgg_yu52c8zXmBxuUk/view?usp=sharing'])
                                    ]),
            '2020-03-13':np.vstack([np.hstack([800,'https://drive.google.com/file/d/1M7R4yf9s6ioJ9bB-tNRUO0sYOO5hNCk1/view?usp=sharing']),
                                    np.hstack([900,'https://drive.google.com/file/d/1vWeuip0TWLo653A7PtcO1y7isLLU6lY2/view?usp=sharing']),
                                    np.hstack([1000,'https://drive.google.com/file/d/1dMqMkBtV-65wx1FGfS2QPFCaPtnXOC4g/view?usp=sharing']),
                                    np.hstack([1100,'https://drive.google.com/file/d/1-IR1w4yPCsXomUTa9qZ_Cwu8o1QQmfNx/view?usp=sharing']),
                                    np.hstack([1200,'https://drive.google.com/file/d/1K-K0RaDpfXiZTqjmygOU9kYAW9KwWADf/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.com/file/d/1pSrMeC9V_OBkKNelCag7ktd_CwpgScJg/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/18JaS0JjbifFz5f2gWhbpkruu97Gf-8ep/view?usp=sharing']),
                                    np.hstack([1500,'https://drive.google.com/file/d/1Mgt1v6z-MwwKE2iAeSP_PvUcFGU9HmSf/view?usp=sharing']),
                                    np.hstack([1600,'https://drive.google.com/file/d/1HqYa1obv6VQstRUP5AYujOdhVj5qCixE/view?usp=sharing']),
                                    np.hstack([1700,'https://drive.google.com/file/d/1V6DSMbB_yasl-yzqbZQqL6OOV1-rY0_1/view?usp=sharing']),
                                    np.hstack([1800,'https://drive.google.com/file/d/10DCmW6NFB-m064e5bGHWjycW3y27U9m7/view?usp=sharing'])
                                    ]),
            '2020-03-14':np.vstack([np.hstack([800,'https://drive.google.com/file/d/1HVi4e71IE_gdipt0mLpjM4s1q2VLDymI/view?usp=sharing']),
                                    np.hstack([900,'https://drive.google.com/file/d/1oOwKMjAGtfGLR65SyFBcpYDrZIsBJ2JH/view?usp=sharing']),
                                    np.hstack([1000,'https://drive.google.com/file/d/1sv6VHASV4rrFhYBKEBHrx8-AxGqcsK7r/view?usp=sharing']),
                                    np.hstack([1100,'https://drive.google.com/file/d/1xQJ7e_55FZ8IKt8c42ItlHtqP4OuzcL6/view?usp=sharing']),
                                    np.hstack([1200,'https://drive.google.com/file/d/1-SQDmdt_MxIYPmN8XEQuruH5yiFuCI2I/view?usp=sharing']),
                                    np.hstack([1300,'https://drive.google.com/file/d/1ibBPg7rXJfhWj8nIzh6mAgM_RYzgzdTj/view?usp=sharing']),
                                    np.hstack([1400,'https://drive.google.com/file/d/1JCFMLPHJ9_ywf-2QKv2BtqisVFOcpw7w/view?usp=sharing']),
                                    np.hstack([1600,'https://drive.google.com/file/d/10HH0L8iGXaE3drB4wB5F-rqAktnTwOLW/view?usp=sharing']),
                                    np.hstack([1700,'https://drive.google.com/file/d/1k0XQ6lT5m7WBXymWeZYV6YZbT5BLU4lT/view?usp=sharing']),
                                    np.hstack([1800,'https://drive.google.com/file/d/1Kcxe9unQ43fQ-8v1YN3m-Vm7OlEVSSsg/view?usp=sharing'])
                                    ])
                                    
    
    }
    
    # Reformat each link array in the dictionary so that its row is its hour, NaNs where videos don't exist #
    for key in links:
        newar = list()
        for ii in range(0,19):
            try:
                newar.append(links[key][np.where(links[key][:,0]==(str(ii)+'00'))[0],1][0])
            except:
                newar.append(np.nan)
        links[key] = newar
    
    
    #=============================================================================#
    # Download all the videos (if they haven't been already) and put the paths in an array #
    #=============================================================================#
    pthList = []
    for key in links:
        ar = links[key]
        for i in range(8,len(ar)):
            try: # If the video exists this try stateent will throw an exception #
                np.isnan(ar[i])
            except:
                ID = ar[i].split('/')[5]
                
                # Make the correct filename by getting the date and time of this video #
                direc = '/Users/matthewconlin/Documents/Research/WebCAT/Applications/Jupiter/RawVideo/'
                
                if i<10:
                    dumhr = '0'
                else:
                    dumhr =''
                    
                fname = 'JupiterCam_'+key.replace('-','')+dumhr+str(i)+'00'
                destination = direc+fname+'.mp4'
               
                # Download and reduce the video the video #
                if not os.path.exists(destination.split('.')[0]+'_reduced.avi'):
                    GrabGDVid(ID,destination)
                    ReduceVid(destination)
                else:
                    pass
                
                pthList.append(destination.split('.')[0]+'_reduced.avi')
                
        return links,pthList

