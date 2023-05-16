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

# Makes sure we have a "test-images" folder
if not os.path.exists("test-images"):
    os.mkdir("test-images")
if not os.path.exists("test-videos"):
    os.mkdir("test-videos")

sleepTime = 5 # How long to sleep between rendering images

try:
    print("loading the compound-ray library")
    eyeRenderer = CDLL(os.path.expanduser("~/compound-ray/build/make/lib/libEyeRenderer3.so"))
    print("Successfully loaded ", eyeRenderer)

    eyeTools.configureFunctions(eyeRenderer)

    eyeRenderer.loadGlTFscene(c_char_p(bytes(os.path.expanduser("~/compound-ray/data/natural-standin-sky3.gltf"), 'utf-8')))

    renderWidth = 400
    renderHeight = 400
    eyeRenderer.setRenderSize(renderWidth,renderHeight)
    eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (renderHeight, renderWidth, 4))

    t = [0.0, 0.0, 1.0];

    # Iterate through a few cameras and do some stuff with them
    for i in range(2):
        video_name = "test-videos/test-video-"+str(i)+".mp4"
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 20, (renderWidth, renderHeight))

        for j in range(240):
            # If the current eye is a compound eye, set the sample rate for it high and take another photo
            if(eyeRenderer.isCompoundEyeActive()):
                #print("This one's a compound eye, let's get a higher sample rate image!")
                eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100);
                renderTime = eyeRenderer.renderFrame() # Render the frame
                #eyeRenderer.saveFrameAs(c_char_p(("test-images/test-image-"+str(i)+"-100samples.ppm").encode()))# Save it
                #Image.fromarray(eyeRenderer.getFramePointer()[::-1,:,:3], "RGB").show() # Show it in PIL (the right way up)
                rgb = eyeRenderer.getFramePointer()[::-1,:,:3]
                bgr = rgb[:, :, ::-1]
                video.write(bgr)

                ## Put it back
                #eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer,ommList)
                #eyeRenderer.renderFrame()
                eyeRenderer.displayFrame()
            else:
                # Actually render the frame
                renderTime = eyeRenderer.renderFrame()
                #print("View from camera '", eyeRenderer.getCurrentCameraName(), " rendered in ", renderTime)

                eyeRenderer.displayFrame() # Display the frame in the renderer

                # Save the frame as a .ppm file directly from the renderer
                #eyeRenderer.saveFrameAs(c_char_p(("test-images/test-image-"+str(i)+".ppm").encode()))

                # Retrieve frame data
                # Note: This data is not owned by Python, and is subject to change
                # with subsequent calls to the renderer so must be deep-copied if
                # you wish for it to persist.
                frameData = eyeRenderer.getFramePointer()
                frameDataRGB = frameData[:,:,:3] # Remove the alpha component
                #print("FrameData type:", type(frameData))
                #print("FrameData:\n",frameData)
                #print("FrameDataRGB:\n",frameDataRGB)

                # Use PIL to display the image (note that it is vertically inverted)
                #img = Image.fromarray(frameDataRGB, "RGB")
                #img.show()

                # Vertically un-invert the array and then display
                rightWayUp = np.flipud(frameDataRGB)
                #rightWayUp = frameDataRGB[::-1,:,:] also works

                #frame = Image.fromarray(rightWayUp, "RGB")
                #frame.show()
                bgr = rightWayUp[:, :, ::-1]
                video.write(bgr)

            #time.sleep(1/30)
            if j <= 120:
                eyeRenderer.translateCameraLocally(t[0], t[1], t[2])
            else:
                eyeRenderer.rotateCameraLocallyAround(3.0 / 360.0 * (2.0 * math.pi), 0, 1.0, 0)
        
        video.release()
        # Change to the next Camera
        eyeRenderer.nextCamera()
        #time.sleep(sleepTime)

    
    input("Press enter to exit...")
    # Finally, stop the eye renderer
    eyeRenderer.stop()
except Exception as e:
    print(e);    