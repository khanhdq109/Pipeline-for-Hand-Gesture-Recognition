"""
  Every support functions for the project!
"""

import os
import cv2
import random

def segment(image, x, y, box_width, box_height, img_width = 128, img_height = 128):
  """
    Segment and resize the image with the given coordinates
  """
  cropped = image[y:y+int(box_height), x:x+int(box_width)]
  cropped = cv2.resize(cropped, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
  return cropped

def main():
  # Randomly select an image from the training set
  index = random.randint(0, 1499)
  # Get the image and annotation file names
  images = os.listdir('../datasets\HAGRID_YOLO\images\\train')
  annotations = os.listdir('../datasets\HAGRID_YOLO\labels\\train')
  # Open the image and annotation
  img = images[index]
  anno = annotations[index]
  img = cv2.imread(os.path.join('../datasets\HAGRID_YOLO\images\\train', img))
  with open(os.path.join('../datasets\HAGRID_YOLO\labels\\train', anno), 'r') as f:
    lines = f.readlines()
    # Get bboxes coordinates
    for line in lines:
      tmp = line.split()
      # Get xmin, ymin, box_with, box_height
      box_width = float(tmp[3]) * 640
      box_height = float(tmp[4]) * 480
      xmin = int(float(tmp[1]) * 640 - box_width / 2)
      ymin = int(float(tmp[2]) * 480 - box_height / 2)
      # Segment the image
      cropped = segment(img, xmin, ymin, box_width, box_height)
      # Show the image
      cv2.imshow('image', cropped)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
