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

import configparser

# # set working dir
# import os
# os.chdir('../')

config = configparser.ConfigParser()
config.read("Scenes/TubeScene/tube_scene.txt")

videoFrames = int(config.get("variables", "videoFrames"))
blenderFile = config.get("variables", "blenderFile")
videoName = config.get("variables", "videoName")

movement_data = config.items("movement")

# Makes sure we have a "TestVideos" folder
if not os.path.exists("Frames/" + videoName + "/data_extraction_frames"):
    os.makedirs("Frames/" + videoName + "/data_extraction_frames")

try:
    # load the compound-ray library
    print("loading the compound-ray library")
    eyeRenderer = CDLL(
        os.path.expanduser("~/compound-ray/build/make/lib/libEyeRenderer3.so")
    )
    print("Successfully loaded ", eyeRenderer)

    # Configure the renderer's function outputs and inputs
    eyeTools.configureFunctions(eyeRenderer)

    # Load the modified example scene
    eyeRenderer.loadGlTFscene(
        c_char_p(
            bytes(
                os.path.expanduser(
                    "~/Documents/GitHub/CompoundRayTests/Scenes/"
                    + videoName
                    + "/"
                    + blenderFile
                ),
                "utf-8",
            )
        )
    )

    # Set the frame size.
    renderWidth = 1551
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth, renderHeight)

    # restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(
        dtype=c_ubyte, shape=(renderHeight, renderWidth, 4)
    )

    video_name = (
        "Frames/" + videoName + "/data_extraction_frames/test_video_" + str(0) + ".mp4"
    )
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        20,
        (renderWidth, renderHeight),
    )

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

    for j in range(videoFrames):

        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100)
        renderTime = eyeRenderer.renderFrame()  # Render the frame

        eyeRenderer.displayFrame()

        rgb = eyeRenderer.getFramePointer()[
            ::, :, :3
        ]  # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)

        ommatid_data = []

        for i in range(1551):
            old_col = rgb[0, i]  # i: an index of ommatidium
            col = np.array(old_col)
            new_col = col.astype(float)
            grey_scale = new_col[0] * 0.299 + new_col[1] * 0.587 + new_col[2] * 0.114
            ommatid_data.append(grey_scale)

        # convert RGB to BGR
        bgr = rgb[:, :, ::-1]
        # write the frame to the output video
        video.write(bgr)

        frame_ommatid_data.append(ommatid_data)

        # Movement function
        for k in range(len(movement_data)):
            if j <= int(movement_data[k][0]):
                eval(movement_data[k][1])
                break

    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()

    ommatid_data = np.array(ommatid_data)
    frame_ommatid_data = np.array(frame_ommatid_data)

    print(ommatid_data)
    print(ommatid_data.shape)

    print(frame_ommatid_data)
    print(frame_ommatid_data.shape)

    with open("MotionDetector/extraction_test.pkl", "wb") as handle:
        pickle.dump(frame_ommatid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ommatid_data = frame100_ommatid_data
    # np.savetxt("DataExtraction/ommatid_data.csv", ommatid_data, delimiter=",")

except Exception as e:
    print(e)
