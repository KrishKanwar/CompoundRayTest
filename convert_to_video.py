import cv2
import os

def images_to_video(folder_path, output_video_path, fps=30):
    # Get list of all image files in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    images.sort()  # Sort the images by name

    if not images:
        print("No JPG images found in the folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")

# Example usage:
images_to_video('OutputData/GeneralScene/CompoundEyeFrames', 'output_video.mp4', fps=30)