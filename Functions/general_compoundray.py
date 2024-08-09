# Produces panoramic and compound-eye videos for a scene
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

config = configparser.ConfigParser()

# MetaTxt to retrieve csv files and scene txt (later replace with a function)
config.read("MetaTxt.txt")
readPath = config.get("data", "path")  # Path to scene information

# Read in scene txt
config.read(readPath)
videoFrames = int(config.get("variables", "videoFrames"))  # number of frames to run
blenderFile = config.get("variables", "blenderFile")  # gltf file
videoName = config.get("variables", "videoName")  # name of scene folder
movement_data = config.items("movement")  # camera movement

# Create folder for video frames and video to be saved to
# Code runs two cameras in this case
if not os.path.exists("OutputData/" + videoName + "/CompoundEyeFrames"):
    os.makedirs("OutputData/" + videoName + "/CompoundEyeFrames")

if not os.path.exists("OutputData/" + videoName + "/PanoramicEyeFrames"):
    os.makedirs("OutputData/" + videoName + "/PanoramicEyeFrames")

try:
    # Load the compound-ray library
    print("loading the compound-ray library")
    eyeRenderer = CDLL(
        os.path.expanduser("~/compound-ray/build/make/lib/libEyeRenderer3.so")
    )
    print("Successfully loaded ", eyeRenderer)

    # Configure the renderer's function outputs and inputs
    # Moves functions from compound-ray library to python
    eyeTools.configureFunctions(eyeRenderer)

    # Load the modified example scene
    # eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser("~/Documents/GitHub/CompoundRayTests/Takashi-Test/Takashi-original-test-scene.gltf"), 'utf-8')))
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
    renderWidth = 400
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth, renderHeight)

    # restype (result type) = RGBA 24bit
    eyeRenderer.getFramePointer.restype = ndpointer(
        dtype=c_ubyte, shape=(renderHeight, renderWidth, 4)
    )

    # Camera 0: regular-panoramic  Camera 1: lens_opticAxis_acceptance.eye
    for i in range(2):
        # Set camera
        if i == 0:
            eyeRenderer.gotoCameraByName(c_char_p(b"regular-panoramic"))
        elif i == 1:
            eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-spherical-projector"))

        for j in range(videoFrames):

            # If the current eye is a compound eye, set the sample rate for it high
            if eyeRenderer.isCompoundEyeActive():
                eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100)
                renderTime = eyeRenderer.renderFrame()  # Render the frame

                # Display the frame in the renderer
                eyeRenderer.displayFrame()

                # Retrieve frame data
                # Note: This data is not owned by Python, and is subject to change
                # with subsequent calls to the renderer so must be deep-copied if
                # you wish for it to persist.
                rgb = eyeRenderer.getFramePointer()[
                    ::-1, :, :3
                ]  # Remove the alpha component and vertically un-invert the array and then display (The retrieved frame data is vertically inverted)

                # Convert RGB to BGR
                bgr = rightWayUp[:, :, ::-1]

                # Write frame
                image_name = (
                    "OutputData/"
                    + videoName
                    + "/CompoundEyeFrames/cef"
                    + str(j)
                    + ".jpg"
                )
                cv2.imwrite(image_name, bgr)

            else:
                # Render the frame
                renderTime = eyeRenderer.renderFrame()
                # Display the frame in the renderer
                eyeRenderer.displayFrame()

                # Retrieve frame data
                frameDataRGB = eyeRenderer.getFramePointer()[
                    :, :, :3
                ]  # Remove the alpha component

                # Vertically un-invert the array and then display (The retrieved frame data is vertically inverted)
                rightWayUp = np.flipud(frameDataRGB)
                # rightWayUp = frameDataRGB[::-1,:,:] also works

                # Convert RGB to BGR
                bgr = rightWayUp[:, :, ::-1]

                # Write frame
                image_name = (
                    "OutputData/"
                    + videoName
                    + "/PanoramicEyeFrames/pef"
                    + str(j)
                    + ".jpg"
                )
                cv2.imwrite(image_name, bgr)

            # Movement function
            for k in range(len(movement_data)):
                if j <= int(movement_data[k][0]):
                    eval(movement_data[k][1])
                    break

    input("Press enter to exit...")

    # Finally, stop the eye renderer
    eyeRenderer.stop()

except Exception as e:
    print(e)
