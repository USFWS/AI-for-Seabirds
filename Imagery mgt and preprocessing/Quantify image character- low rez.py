import cv2
import glob
import os
import numpy
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Input: files = directory of images, export_csv = csv that reports brightness, contrast for each image
files = glob.glob("C:/BP/Annotation_analysis/cvat_images_good/*.jpg")

export_csv = "C:/BP/Annotation_analysis/new_character.csv"

# orig pixels= 6464 * 4848
# Input desired pixel dimensions; e.g., 1/16 size of orig pixels
width_px = 404
height_px = 303

total_pixels = width_px * height_px

image_list = []
contrast_sd_list = []
saturation_percent_list = []
contrast10_list = []

x=0
for file in files:
    x= x+1
    print(x)
    file1 = os.path.basename(file)
    print(file1)
    img = cv2.imread(file)
    img = cv2.resize(img, dsize = (width_px, height_px), interpolation = cv2.INTER_AREA)

    image_list.append(file1)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast sd
    contrast_sd =numpy.std(img_grey)
   # print("Contrast sd: ", contrast_sd)
    contrast_sd_list.append(contrast_sd)

    # Calculate % saturation with global thresholding: INPUT THRESHOLD VALUE,
    # If pixel value is greater than a threshold value, it is assigned one value
    ret1, th1 = cv2.threshold(img_grey, 230, 1, cv2.THRESH_BINARY)
    sum1 = numpy.sum(th1)

    saturation_prop = sum1/ total_pixels
    saturation_percent = saturation_prop*100
    saturation_percent_list.append(saturation_percent)
   # print("saturation: ", saturation_percent)

    # Texture analysis
    glcm = graycomatrix(img_grey, distances=[1], angles=[45], levels=256, symmetric=True, normed=True)

    # Texture contast
   # contrast10 = graycoprops(glcm, 'contrast')
    #contrast10_list.append(contrast10)

pd.DataFrame({"unique_image": image_list, "contrast_sd": contrast_sd_list, "percent_saturation": saturation_percent_list}).to_csv(export_csv, index=True)
