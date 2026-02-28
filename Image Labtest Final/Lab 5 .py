#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 02:07:25 2025

@author: mahirazmainhaque
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# take input
img_input = cv2.imread('two_noise.jpeg', 0)
img = img_input.copy()

def butterworth(img,uk_list, vk_list, D0, n):
    M, N = img.shape
    H = np.ones((M, N), dtype=np.float32)
    for k in range(len(uk_list)):
        uk = uk_list[k]
        vk = vk_list[k]
        for u in range(M):
            for v in range(N):
                Dk = np.sqrt((u - N/2 - uk)**2 + (v - M/2 - vk)**2)
                D_k = np.sqrt((u - N/2 + uk)**2 + (v - M/2 + vk)**2)

                if Dk == 0:
                    Dk = 1e-6
                if D_k == 0:
                    D_k = 1e-6

                # Butterworth notch reject formula
                Hk = 1 / (1 + (D0 / Dk)**(2 * n))
                H_k = 1 / (1 + (D0 / D_k)**(2 * n))
                H[v, u] = H[v, u] * Hk * H_k

    
    return H


def butterworth2(img,rk_list, ck_list, D0, n):
    M, N = img.shape
    H = np.ones((M, N), dtype=np.float32)
    for k in range(len(rk_list)):
        rk = rk_list[k]
        ck = ck_list[k]
        for u in range(M):
            for v in range(N):
                rk_ = M - rk
                ck_ = N - ck
                Dk = np.sqrt((u - rk)**2 + (v - ck)**2)
                D_k = np.sqrt((u - rk_)**2 + (v - ck_)**2)

                if Dk == 0:
                    Dk = 1e-6
                if D_k == 0:
                    D_k = 1e-6

                # Butterworth notch reject formula
                Hk = 1 / (1 + (D0 / Dk)**(2 * n))
                H_k = 1 / (1 + (D0 / D_k)**(2 * n))
                H[u, v] = H[u, v] * Hk * H_k

    
    return H


def notchReject(img,uk_list, vk_list, D0):
    M, N = img.shape
    H = np.ones((M, N), dtype=np.float32)
    for k in range(len(uk_list)):
        uk = uk_list[k]
        vk = vk_list[k]
        for u in range(M):
            for v in range(N):
                Dk = np.sqrt((u - N/2 - uk)**2 + (v - M/2 - vk)**2)
                D_k = np.sqrt((u - N/2 + uk)**2 + (v - M/2 + vk)**2)
                if Dk <= D0 or D_k <= D0:
                    H[v, u] = 0

    return H

noise_points = [(272,256),(261,261)]

uk_list = []
vk_list = []

for point in noise_points:
    t1 = abs(256 - point[0])
    t2 = abs(256 - point[1])
    uk_list.append(t1)
    vk_list.append(t2)
    
D0 = 5
n = 2
ck_list = [272,261]
rk_list = [256,261]

H1 = butterworth(img,uk_list,vk_list,D0,n)
H2 = notchReject(img,uk_list,vk_list,D0)
H3 = butterworth2(img,rk_list,ck_list,D0,n)

# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
#ft_shift = ft
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

magnitude_spectrum_ac=magnitude_spectrum_ac*H1
magnitude_spectrum_denoised = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
## phase add F(u,v)=∣F(u,v)∣*e^jθ(u,v)
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))
# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

plt.figure(figsize=(16,8))
plt.suptitle('Removing frequency at (272,256) & (261,261) , D0 = 5, n=2')
plt.subplot(2,3,1)
plt.imshow(img,cmap='gray')
plt.title('Figure1: Original Noisy Image')
plt.axis('off') 

plt.subplot(2,3,2)
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Figure2: Magnitude Spectrum')
plt.axis('off') 

plt.subplot(2,3,3)
plt.imshow(ang_,cmap='gray')
plt.title('Figure3: Phase')
plt.axis('off') 

plt.subplot(2,3,4)
plt.imshow(H3,cmap='gray')
plt.title('Figure4: Butterworth filter')
plt.axis('off') 

plt.subplot(2,3,5)
plt.imshow(magnitude_spectrum_denoised,cmap='gray')
plt.title('Figure5: FFT of after applying butterworth notch filter')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(img_back_scaled,cmap='gray')
plt.title('Figure6: Reconstructed Denoised Image')
plt.axis('off') 
plt.savefig('1.png',dpi=300)
plt.show()