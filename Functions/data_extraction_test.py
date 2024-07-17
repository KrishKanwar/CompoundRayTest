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

config = configparser.ConfigParser()

# MetaTxt to retrieve csv files and scene txt (later replace with a function)
config.read("MetaTxt.txt")
readPath = config.get("data", "path")
csvData = config.get("data", "csvData")
# direction = int(config.get("data", "direction"))

data = np.genfromtxt(csvData, delimiter=",")
num_omm = data.shape[0] - 1  # get number of ommatidia in complete eye (l & r)

# Read in scene txt
config.read(readPath)
videoFrames = int(config.get("variables", "videoFrames"))  # number of frames to run
blenderFile = config.get("variables", "blenderFile")  # gltf file
videoName = config.get("variables", "videoName")  # name of scene folder
movement_data = config.items("movement")  # camera movement

# Makes sure we have a "TestVideos" folder
if not os.path.exists("OutputData/" + videoName + "/DataExtractionFrames"):
    os.makedirs("OutputData/" + videoName + "/DataExtractionFrames")

try:
    # Load the compound-ray library
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
    renderWidth = num_omm
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth, renderHeight)

    # restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(
        dtype=c_ubyte, shape=(renderHeight, renderWidth, 4)
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

    ommatid_data = []

    for j in range(videoFrames):

        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100)
        renderTime = eyeRenderer.renderFrame()  # Render the frame

        eyeRenderer.displayFrame()

        rgb = eyeRenderer.getFramePointer()[
            ::, :, :3
        ]  # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)

        single_frame_ommatid_data = []

        # Get color of every ommatidia for frame and apply greyscale
        for i in range(num_omm):
            old_col = rgb[0, i]  # i: an index of ommatidium
            col = np.array(old_col)
            new_col = col.astype(float)
            grey_scale = new_col[0] * 0.299 + new_col[1] * 0.587 + new_col[2] * 0.114
            single_frame_ommatid_data.append(grey_scale)

        # Convert RGB to BGR
        bgr = rgb[:, :, ::-1]

        # Write frame
        image_name = (
            "OutputData/" + videoName + "/DataExtractionFrames/def" + str(j) + ".jpg"
        )
        cv2.imwrite(image_name, bgr)

        ommatid_data.append(single_frame_ommatid_data)

        # Movement function
        for k in range(len(movement_data)):
            if j <= int(movement_data[k][0]):
                eval(movement_data[k][1])
                break

    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()

    # Save ommatid data
    ommatid_data = np.array(ommatid_data)
    print(ommatid_data.shape)
    with open("OutputData/" + videoName + "/i_de.pkl", "wb") as handle:
        pickle.dump(ommatid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

except Exception as e:
    print(e)
