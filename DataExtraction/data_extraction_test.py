import os.path
import time
import math
from ctypes import *
from sys import platform
from numpy.ctypeslib import ndpointer
import numpy as np

from PIL import Image
import cv2

import pickle

from threading import Timer

import eye_renderer_helper_functions as eyeTools

# Makes sure we have a "TestVideos" folder
if not os.path.exists("DataExtraction/TestVideos"):
    os.mkdir("DataExtraction/TestVideos")

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
        "~/Documents/GitHub/CompoundRayTests/DataExtraction/validation_test_smaller_extraction.gltf"), 'utf-8')))
    
    #eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser(
    #        "~/Documents/GitHub/CompoundRayTests/Takashi-Test/Takashi-original-test-scene.gltf"), 'utf-8')))
    
    # Set the frame size.
    renderWidth = 1551
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth, renderHeight)

    # restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(
        dtype=c_ubyte, shape=(renderHeight, renderWidth, 4))
    
    video_name = "DataExtraction/TestVideos/test_video_"+str(0)+".mp4"
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 20, (renderWidth, renderHeight))

    # Switch to insect-eye-fast-vector camera (extracts data)
    eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-fast-vector"))

    # Commented out code that stretches out to fill the data to the entire screen

    # # Prepare to generate vector data (1000 ommatidia)
    # vectorWidth = 1000
    # # The upper bound of how many samples will be taken per ommatidium in the analysis
    # maxOmmatidialSamples = renderWidth
    # spreadSampleCount = 1000  # How many times each frame is rendered to get a sense of the spread of results from a given ommatidium at different sampling rates
    # eyeTools.setRenderSize(eyeRenderer, vectorWidth, 1)

    frame_ommatid_data = []
    frame100_ommatid_data = []

    for j in range(300):

        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100)
        renderTime = eyeRenderer.renderFrame()  # Render the frame

        eyeRenderer.displayFrame()

        rgb = eyeRenderer.getFramePointer()[::,:,:3] # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)
        
        ommatid_data = []

        for i in range(1551):
            old_col = rgb[0,i] # i: an index of ommatidium
            col = np.array(old_col)
            new_col = col.astype(float)
            grey_scale = new_col[0]*0.299 + new_col[1]*0.587 + new_col[2]*0.114
            ommatid_data.append(grey_scale)

        #convert RGB to BGR
        bgr = rgb[:, :, ::-1]
        #write the frame to the output video
        video.write(bgr)

        frame_ommatid_data.append(ommatid_data)

        if (j == 90):
            frame100_ommatid_data = ommatid_data

        # i = 0
        # rgb = eyeRenderer.getFramePointer()[:, :, :3]
        # col = rgb[0, i]  # i: an index of ommatidium

        if j <= 300:
            eyeRenderer.translateCameraLocally(
                0.0, 0.0, 0.2)  # move forward (0-120 frame)
        else:
            # rotate 360 degree along y axis (120-240 frame)
            eyeRenderer.rotateCameraLocallyAround(
                3.0 / 360.0 * (2.0 * math.pi), 0, 1.0, 0)

    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()

    ommatid_data = np.array(ommatid_data)
    frame_ommatid_data = np.array(frame_ommatid_data)

    print(ommatid_data)
    print(ommatid_data.shape)
    
    print(frame_ommatid_data)
    print(frame_ommatid_data.shape)

    # with open('DataExtractionTest/extraction_test.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(frame_ommatid_data, f) # serialize the list
    #     f.close()

    # a = {'hello': 'world'}

    with open('MotionDetector/extraction_test.pkl', 'wb') as handle:
        pickle.dump(frame_ommatid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('MotionDetector/extraction_test.pkl', 'rb') as handle:
    #     b = pickle.load(handle)

    # print(frame_ommatid_data == b)

    ommatid_data = frame100_ommatid_data
    np.savetxt("DataExtraction/ommatid_data.csv", ommatid_data, delimiter=",")

except Exception as e:
    print(e)
