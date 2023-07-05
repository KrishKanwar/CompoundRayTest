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

try:
  eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-fast-vector"))

  # Prepare to generate vector data (1000 ommatidia)
  vectorWidth = 1000
  maxOmmatidialSamples = renderWidth # The upper bound of how many samples will be taken per ommatidium in the analysis
  spreadSampleCount = 1000 # How many times each frame is rendered to get a sense of the spread of results from a given ommatidium at different sampling rates
  eyeTools.setRenderSize(eyeRenderer, vectorWidth, 1)

  # Create a numpy array to store the eye data
  # This is a set of eye matricies, each one being a 1st-order stack of samples (the width of the number of ommatidia, and 3 channels deep)
  eyeSampleMatrix = np.zeros((maxOmmatidialSamples,spreadSampleCount, vectorWidth, 3), dtype=np.uint8)

  # Iterate over eye sample counts
  for idx, samples in enumerate(range(1, maxOmmatidialSamples+1)):
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
    eyeRenderer.renderFrame() # First call to ensure randoms are configured

    # For each sample count, generate N images to compare
    for i in range(spreadSampleCount):
      renderTime = eyeRenderer.renderFrame() # Second call to actually render the image

      # Retrieve the data
      frameData = eyeRenderer.getFramePointer()
      frameDataRGB = frameData[:,:,:3] # Remove the alpha channel
      eyeSampleMatrix[idx,i,:,:] = np.copy(frameDataRGB[:, :, :])

    eyeRenderer.displayFrame()

except Exception as e:
    print(e);    
