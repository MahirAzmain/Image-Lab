#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 21:59:53 2025

@author: mahirazmainhaque
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Gaussian function
def gaussian_function(x, y, sigma):
    return (1 / (2 * math.pi * (sigma**2))) * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

def gaussian_kernel_function(size, sigma):
    kernel = np.zeros((size, size))
    k = size // 2

    for i in range(-k, k + 1): #x
        for j in range(-k, k + 1): #y
            g = gaussian_function(i, j, sigma)
            kernel[k - j, k + i] = g
    
    #kernel /= np.sum(kernel)
    return kernel


# X-derivative of Gaussian kernel
def x_derivative_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float64)
    k = size // 2

    for i in range(-k, k + 1):  # x
        for j in range(-k, k + 1):  # y
            g = gaussian_function(i, j, sigma)
            kernel[k - j, k + i] = (-i / (sigma**2)) * g  # Correct indexing

    return kernel

# Y-derivative of Gaussian kernel
def y_derivative_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float64)
    k = size // 2

    for i in range(-k, k + 1):  # x
        for j in range(-k, k + 1):  # y
            g = gaussian_function(i, j, sigma)
            kernel[k - j, k + i] = (-j / (sigma**2)) * g  # Correct indexing

    return kernel


# Laplacian of Gaussian sharpening function
def log_function(u, v, sigma):
    factor = (u**2 + v**2 - 2*sigma**2)/sigma**4
    return factor * gaussian_function(u, v, sigma)

# Gaussian sharpening kernel
def gaussian_sharpening_kernel(size, sigma):
    size = int(size)|1
    kernel = np.zeros((size, size))
    k = size // 2
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            kernel[k-j, k+i] = log_function(i, j, sigma)
            
    # Zero mean for sharpening
    kernel -= np.mean(kernel)
    return kernel


def double_thresholding(grad_img, T_high,T_low):
    
    grad_magnitude = cv2.normalize(grad_img, None, 0, 255, cv2.NORM_MINMAX)
    grad_magnitude = grad_magnitude.astype(np.uint8)
    
    out = grad_magnitude.copy()
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i][j]>T_high:
                out[i][j] = 255
            elif out[i][j]>T_low:
                out[i][j] = 128
            else:
                out[i][j] = 0
            
    return out


def zero_crossing(log_img):
    M,N = log_img.shape
    
    zero_crossed = np.zeros((M,N))
    zero_strength = np.zeros((M,N))
    
    for i in range (1,M-1):
        for j in range(1,N-1):
            if(log_img[i][j]*log_img[i+1][j]<0 or log_img[i][j]*log_img[i-1][j]<0 or log_img[i][j]*log_img[i][j+1]<0 or log_img[i][j]*log_img[i][j-1]<0 ):
                zero_crossed[i][j] = log_img[i][j]
                zero_strength[i][j] =  np.abs(log_img[i][j]-log_img[i+1][j])+ np.abs(log_img[i][j]-log_img[i-1][j]) + np.abs(log_img[i][j]-log_img[i][j+1]) + np.abs(log_img[i][j]-log_img[i][j-1]) 
            else: 
                zero_crossed[i][j] = 0
                zero_strength[i][j] = 0
    return (zero_crossed,zero_strength)

def threshold_zs(zero_strength,Th):
    out = zero_strength.copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i][j]>Th:
                out[i][j] = 255
            else:
                out[i][j] = 0
            
    return out

sigma = 1.0

x_kernel = x_derivative_gaussian_kernel(int(7*sigma)|1,sigma)
y_kernel = y_derivative_gaussian_kernel(int(7*sigma)|1,sigma)


building_img = cv2.imread('buildings.jpg',0)

dx_building = cv2.filter2D(building_img.astype(np.float32),-1,np.flipud(np.fliplr(x_kernel)))

dy_building = cv2.filter2D(building_img.astype(np.float32),-1,np.flipud(np.fliplr(y_kernel)))

mag_building = cv2.magnitude(dx_building.astype(np.float32), dy_building.astype(np.float32))

out = double_thresholding(mag_building,150,100)

plt.subplot(1,3,1)
plt.imshow(building_img,cmap="gray")
plt.title("Fig 3: Original Image")

plt.subplot(1,3,2)
plt.imshow(dx_building,cmap="gray")
plt.title("Fig 3: Dx Image")

plt.subplot(1,3,3)
plt.imshow(dy_building,cmap="gray")
plt.title("Fig 3: Dy Image")
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.imshow(mag_building,cmap="gray")
plt.title("Fig 3: Magnitude Image")


plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.imshow(out,cmap="gray")
plt.title("Fig 3: Double thresholding Image")

plt.show()



def add_gaussian_noise(img, mean=0, sigma=1):
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    
    # add noise and clip values
    noisy_img = img.astype(np.float32) + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


Lena = cv2.imread('Lena.jpg',0)


Lena = add_gaussian_noise(Lena)
log_kernel = gaussian_sharpening_kernel((9*1)|1,1)
log_Lena = cv2.filter2D(Lena.astype(np.float32),-1,np.flipud(np.fliplr(log_kernel)))

zc,zs = zero_crossing(log_Lena)
edge_lena = threshold_zs(zs, 20)


plt.subplot(1,3,1)
plt.imshow(cv2.normalize(log_Lena,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cmap="gray")
plt.title("Fig 3: Log Image")

plt.subplot(1,3,2)
plt.imshow(cv2.normalize(zc,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cmap="gray")
plt.title("Fig 3: ZC")

plt.subplot(1,3,3)
plt.imshow(cv2.normalize(zs,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cmap="gray")
plt.title("Fig 3: ZS")
plt.show()

plt.figure(figsize=(10, 4))
plt.imshow(edge_lena,cmap="gray")
plt.title("Fig 3: Double thresholding Image")

plt.show()
