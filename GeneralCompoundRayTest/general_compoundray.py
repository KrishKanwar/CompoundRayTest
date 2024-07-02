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

import eye_renderer_helper_functions as eyeTools

import configparser

# # set working dir
# import os
# os.chdir('../')

config = configparser.ConfigParser()
config.read("GeneralCompoundRayTest/Scenes/GeneralScene/general_scene.txt")

videoFrames = int(config.get("variables", "videoFrames"))
blenderFile = config.get("variables", "blenderFile")
videoName = config.get("variables", "videoName")

movement_data = config.items("movement")

# Create folder for video frames and video to be saved to
if not os.path.exists("GeneralCompoundRayTest/Scenes/" + videoName + "/VideoFrames"):
    os.makedirs("GeneralCompoundRayTest/Scenes/" + videoName + "/VideoFrames")

try:
    #load the compound-ray library
    print("loading the compound-ray library")
    eyeRenderer = CDLL(os.path.expanduser("~/compound-ray/build/make/lib/libEyeRenderer3.so"))
    print("Successfully loaded ", eyeRenderer)

    #Configure the renderer's function outputs and inputs
    eyeTools.configureFunctions(eyeRenderer)

    #Load the modified example scene
    # eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser("~/Documents/GitHub/CompoundRayTests/Takashi-Test/Takashi-original-test-scene.gltf"), 'utf-8')))
    eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser("~/Documents/GitHub/CompoundRayTests/GeneralCompoundRayTest/Scenes/" + videoName + "/" + blenderFile), 'utf-8')))

    #Set the frame size.
    renderWidth = 400
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth,renderHeight)
    #restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (renderHeight, renderWidth, 4))

    #Camera 0: regular-panoramic  Camera 1: lens_opticAxis_acceptance.eye
    for i in range(2):
        for j in range(videoFrames):
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
                bgr = rightWayUp[:, :, ::-1]
                #write the frame to the output video
                image_name = "GeneralCompoundRayTest/Scenes/" + videoName + "/VideoFrames/compound_eye_frame"+str(j)+".jpg"
                cv2.imwrite(image_name, bgr)

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
                image_name = "GeneralCompoundRayTest/Scenes/" + videoName + "/VideoFrames/panoramic_eye_frame"+str(j)+".jpg"
                cv2.imwrite(image_name, bgr)
            
            # Movement function
            for k in range(len(movement_data)):
                if j<=int(movement_data[k][0]):
                    eval(movement_data[k][1])
                    break

        # Change to the next Camera
        eyeRenderer.nextCamera()
    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()
    
except Exception as e:
    print(e);    
