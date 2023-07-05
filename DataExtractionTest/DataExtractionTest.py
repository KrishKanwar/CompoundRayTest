import os.path
import time
import math
from ctypes import *
from sys import platform
from numpy.ctypeslib import ndpointer
import numpy as np

from PIL import Image
import cv2

from threading import Timer

import eyeRendererHelperFunctions as eyeTools

# Makes sure we have a "test-videos" folder
if not os.path.exists("DataExtractionTest/test-videos"):
    os.mkdir("DataExtractionTest/test-videos")

try:
    #load the compound-ray library
    print("loading the compound-ray library")
    eyeRenderer = CDLL(os.path.expanduser("~/compound-ray/build/make/lib/libEyeRenderer3.so"))
    print("Successfully loaded ", eyeRenderer)

    #Configure the renderer's function outputs and inputs
    eyeTools.configureFunctions(eyeRenderer)

    #Load the modified example scene
    eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser("~/Documents/GitHub/CompoundRayTests/DataExtractionTest/DataExtractionTest.gltf"), 'utf-8')))

    #Set the frame size.
    renderWidth = 400
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth,renderHeight)
    #restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (renderHeight, renderWidth, 4))

    #Camera 0: regular-panoramic  Camera 1: lens_opticAxis_acceptance.eye
    for i in range(2):
        #initialize the opencv video writer
        video_name = "DataExtractionTest/test-videos/test-video-"+str(i)+".mp4"
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 40, (renderWidth, renderHeight))

        for j in range(200):
            # If the current eye is a compound eye, set the sample rate for it high
            if(eyeRenderer.isCompoundEyeActive()):
                eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100);
                renderTime = eyeRenderer.renderFrame() # Render the frame
                #display the frame in the renderer
                eyeRenderer.displayFrame()

                # Retrieve frame data
                # Note: This data is not owned by Python, and is subject to change
                # with subsequent calls to the renderer so must be deep-copied if
                # you wish for it to persist.
                rgb = eyeRenderer.getFramePointer()[::-1,:,:3] # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)
                #convert RGB to BGR
                bgr = rgb[:, :, ::-1]
                #write the frame to the output video
                video.write(bgr)
                
            else:
                #Render the frame
                renderTime = eyeRenderer.renderFrame()
                #display the frame in the renderer
                eyeRenderer.displayFrame() 

                # Retrieve frame data
                frameDataRGB = eyeRenderer.getFramePointer()[:,:,:3] # Remove the alpha component

                # vertically un-invert the array and then display (The retrieved frame data is vertically inverted)
                rightWayUp = np.flipud(frameDataRGB)
                #rightWayUp = frameDataRGB[::-1,:,:] also works

                #convert RGB to BGR
                bgr = rightWayUp[:, :, ::-1]
                #write the frame to the output video
                video.write(bgr)
            
            if j <= 50:
                eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, 1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)
            elif j<=100:
                eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, 1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)
            elif j<= 150:
                eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, -1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)
            elif j<= 200:
                eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, -1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)

            # if j <= 20:
            #     eyeRenderer.translateCameraLocally(0.0, 0.5, 0.0) #move forward (0-120 frame)
            # elif j<=100:
            #     eyeRenderer.translateCameraLocally(0.0, 0.0, 1.5) #move forward (0-120 frame)
            # elif j<= 150:
            #     eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, 1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)
            # elif j<= 200:
            #     eyeRenderer.rotateCameraLocallyAround(3.6 / 360.0 * (2.0 * math.pi), 0.0, -1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)
            # else:
            #     eyeRenderer.translateCameraLocally(0.0, 0.0, 1.5) #move forward (0-120 frame) 

            #if j <= 20:
                #eyeRenderer.translateCameraLocally(0.0, 0.0, 0.5) #move forward (0-120 frame)
                #eyeRenderer.translateCameraLocally(0.0, 0.5, 0.0) #move forward (0-120 frame)

            #elif j<=40:
                #eyeRenderer.rotateCameraLocallyAround(-0.5 / 360.0 * (2.0 * math.pi), 1.0, 0, 0) # rotate 360 degree along y axis (120-240 frame)
            
            #elif j>20 & j<=40:
                #eyeRenderer.translateCameraLocally(0.0, 0.0, 0.5) #move forward (0-120 frame)

            #else:
                #eyeRenderer.rotateCameraLocallyAround(5.0 / 360.0 * (2.0 * math.pi), 0.0, 1.0, 0.0) # rotate 360 degree along y axis (120-240 frame)

            # to produce other video for comparison, use this movement instead
            #if j <= 100:
                #eyeRenderer.rotateCameraLocallyAround(-3 / 360.0 * (2.0 * math.pi), 0, 1.0, 0) # rotate 360 degree along y axis (120-240 frame)

        video.release()
        # Change to the next Camera
        eyeRenderer.nextCamera()
    
    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()
except Exception as e:
    print(e);    


