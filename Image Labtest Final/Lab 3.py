#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 22:16:33 2025

@author: mahirazmainhaque
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_manual(channel_data):
    height, width = channel_data.shape
    levels = 256

    hist_values = cv2.calcHist([channel_data],[0],None,[levels],[0,levels]).flatten()
    total_pixels = height * width

    pdf_values = hist_values / total_pixels
    cdf_values = np.cumsum(pdf_values)
    mapping = np.round((levels - 1) * cdf_values).astype(np.uint8)

    equalized_channel = np.zeros_like(channel_data)
    for i in range(height):
        for j in range(width):
            old_pixel = channel_data[i, j]
            equalized_channel[i, j] = mapping[old_pixel]

    eq_hist_values = cv2.calcHist([equalized_channel],[0],None,[levels],[0,levels]).flatten()
    eq_pdf_values = eq_hist_values / total_pixels
    eq_cdf_values = np.cumsum(eq_pdf_values)

    return equalized_channel, hist_values, pdf_values, cdf_values, mapping, eq_hist_values, eq_pdf_values, eq_cdf_values

img = cv2.imread("color_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


red, green, blue = cv2.split(img)
red_eq, red_hist, red_pdf, red_cdf, red_map, red_eq_hist, red_eq_pdf, red_eq_cdf = equalize_histogram_manual(red)
green_eq, green_hist, green_pdf, green_cdf, green_map, green_eq_hist, green_eq_pdf, green_eq_cdf = equalize_histogram_manual(green)
blue_eq, blue_hist, blue_pdf, blue_cdf, blue_map, blue_eq_hist, blue_eq_pdf, blue_eq_cdf = equalize_histogram_manual(blue)
rgb_eq_img = cv2.merge((red_eq, green_eq, blue_eq))

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h_channel, s_channel, v_channel = cv2.split(hsv_img)
v_eq, v_hist, v_pdf, v_cdf, v_map, v_eq_hist, v_eq_pdf, v_eq_cdf = equalize_histogram_manual(v_channel)
hsv_eq_img = cv2.merge((h_channel, s_channel, v_eq))
hsv_eq_rgb = cv2.cvtColor(hsv_eq_img, cv2.COLOR_HSV2RGB)



plt.figure(figsize=(20,12))

plt.subplot(3,4,1)
plt.bar(range(256), red_hist, color='red', alpha=.5)
plt.bar(range(256), green_hist, color='green', alpha=.5)
plt.bar(range(256), blue_hist, color='blue', alpha=.5)
plt.bar(range(256), v_hist, color='pink', alpha=.3)
plt.title("Original Histogram")

plt.subplot(3,4,2)
plt.bar(range(256), red_eq_hist, color='red', alpha=.5)
plt.bar(range(256), green_eq_hist, color='green', alpha=.5)
plt.bar(range(256), blue_eq_hist, color='blue', alpha=.5)
plt.title("Equalized RGB Histogram")

plt.subplot(3,4,3)
plt.bar(range(256), v_eq_hist, color='pink', alpha=.7)
plt.title("Equalized HSV-V Histogram")

plt.subplot(3,4,4)
plt.plot(red_pdf, color='red')
plt.plot(green_pdf, color='green')
plt.plot(blue_pdf, color='blue')
plt.plot(v_pdf, color='pink')
plt.title("Original PDF")

plt.subplot(3,4,5)
plt.plot(red_eq_pdf, color='red')
plt.plot(green_eq_pdf, color='green')
plt.plot(blue_eq_pdf, color='blue')
plt.title("Equalized RGB PDF")

plt.subplot(3,4,6)
plt.plot(v_eq_pdf, color='pink')
plt.title("Equalized HSV-V PDF")

plt.subplot(3,4,7)
plt.plot(red_cdf, color='red')
plt.plot(green_cdf, color='green')
plt.plot(blue_cdf, color='blue')
plt.plot(v_cdf, color='pink')
plt.title("Original CDF")

plt.subplot(3,4,8)
plt.plot(red_eq_cdf, color='red')
plt.plot(green_eq_cdf, color='green')
plt.plot(blue_eq_cdf, color='blue')
plt.title("Equalized RGB CDF")

plt.subplot(3,4,9)
plt.plot(v_eq_cdf, color='pink')
plt.title("Equalized HSV-V CDF")

plt.tight_layout()
plt.show()



# Assuming img, rgb_eq_img, hsv_eq_rgb, red, green, blue, red_eq, green_eq, blue_eq are defined

# Convert BGR to RGB if needed (OpenCV loads images as BGR by default)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(3, 3, figsize=(15, 10))

# Original and equalized RGB
axs[0, 0].imshow(img)
axs[0, 0].set_title("Original RGB")
axs[0, 0].axis('off')

axs[0, 1].imshow(rgb_eq_img)
axs[0, 1].set_title("Equalized RGB")
axs[0, 1].axis('off')

axs[0, 2].imshow(hsv_eq_rgb)
axs[0, 2].set_title("Equalized HSV (V)")
axs[0, 2].axis('off')

# RGB channels
axs[1, 0].imshow(red, cmap='Reds')
axs[1, 0].set_title("Red Channel")
axs[1, 0].axis('off')

axs[1, 1].imshow(green, cmap='Greens')
axs[1, 1].set_title("Green Channel")
axs[1, 1].axis('off')

axs[1, 2].imshow(blue, cmap='Blues')
axs[1, 2].set_title("Blue Channel")
axs[1, 2].axis('off')

# Equalized RGB channels
axs[2, 0].imshow(red_eq, cmap='Reds')
axs[2, 0].set_title("Equalized Red")
axs[2, 0].axis('off')

axs[2, 1].imshow(green_eq, cmap='Greens')
axs[2, 1].set_title("Equalized Green")
axs[2, 1].axis('off')

axs[2, 2].imshow(blue_eq, cmap='Blues')
axs[2, 2].set_title("Equalized Blue")
axs[2, 2].axis('off')

plt.tight_layout()
plt.show()



# Assuming you already have these arrays defined:
# red_pdf, green_pdf, blue_pdf
# red_cdf, green_cdf, blue_cdf

fig, axs = plt.subplots(2, 3, figsize=(18, 8))

# --- PDFs ---
axs[0, 0].plot(red_pdf, color='red')
axs[0, 0].set_title("Red PDF")
axs[0, 0].set_xlim([0, 255])

axs[0, 1].plot(green_pdf, color='green')
axs[0, 1].set_title("Green PDF")
axs[0, 1].set_xlim([0, 255])

axs[0, 2].plot(blue_pdf, color='blue')
axs[0, 2].set_title("Blue PDF")
axs[0, 2].set_xlim([0, 255])

# --- CDFs ---
axs[1, 0].plot(red_cdf, color='red')
axs[1, 0].set_title("Red CDF")
axs[1, 0].set_xlim([0, 255])

axs[1, 1].plot(green_cdf, color='green')
axs[1, 1].set_title("Green CDF")
axs[1, 1].set_xlim([0, 255])

axs[1, 2].plot(blue_cdf, color='blue')
axs[1, 2].set_title("Blue CDF")
axs[1, 2].set_xlim([0, 255])

plt.tight_layout()
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

