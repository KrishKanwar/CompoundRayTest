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
    # load the compound-ray library
    print("loading the compound-ray library")
    eyeRenderer = CDLL(os.path.expanduser(
        "~/compound-ray/build/make/lib/libEyeRenderer3.so"))
    print("Successfully loaded ", eyeRenderer)

    # Configure the renderer's function outputs and inputs
    eyeTools.configureFunctions(eyeRenderer)

    # Load the modified example scene
    eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser(
        "~/Documents/GitHub/CompoundRayTests/DataExtractionTest/DataExtractionTest.gltf"), 'utf-8')))

    # Set the frame size.
    renderWidth = 400
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth, renderHeight)

    # restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(
        dtype=c_ubyte, shape=(renderHeight, renderWidth, 4))
    
    video_name = "DataExtractionTest/test-videos/test-video-"+str(0)+".mp4"
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 20, (renderWidth, renderHeight))

    eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-fast-vector"))

    # Commented out code that stretches out to fill the data to the entire screen

    # # Prepare to generate vector data (1000 ommatidia)
    # vectorWidth = 1000
    # # The upper bound of how many samples will be taken per ommatidium in the analysis
    # maxOmmatidialSamples = renderWidth
    # spreadSampleCount = 1000  # How many times each frame is rendered to get a sense of the spread of results from a given ommatidium at different sampling rates
    # eyeTools.setRenderSize(eyeRenderer, vectorWidth, 1)

    for j in range(240):

        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100)
        renderTime = eyeRenderer.renderFrame()  # Render the frame

        eyeRenderer.displayFrame()

        rgb = eyeRenderer.getFramePointer()[::,:,:3] # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)
        
        ommatid_data = []

        for i in range(300):
            col = rgb[0,i] # i: an index of ommatidium
            ommatid_data.append(col)

        #convert RGB to BGR
        bgr = rgb[:, :, ::-1]
        #write the frame to the output video
        video.write(bgr)

        # i = 0
        # rgb = eyeRenderer.getFramePointer()[:, :, :3]
        # col = rgb[0, i]  # i: an index of ommatidium

        if j <= 120:
            eyeRenderer.translateCameraLocally(
                0.0, 0.0, 1.0)  # move forward (0-120 frame)
        else:
            # rotate 360 degree along y axis (120-240 frame)
            eyeRenderer.rotateCameraLocallyAround(
                3.0 / 360.0 * (2.0 * math.pi), 0, 1.0, 0)

    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()

    print(ommatid_data)

except Exception as e:
    print(e)
