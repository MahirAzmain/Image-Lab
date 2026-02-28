# # Gx_sobel = np.array([
# #     [-1, 0, 1],
# #     [-2, 0, 2],
# #     [-1, 0, 1]
# # ], dtype=np.float32)

# # Gy_sobel = np.array([
# #     [-1, -2, -1],
# #     [ 0,  0,  0],
# #     [ 1,  2,  1]
# # ], dtype=np.float32)



import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def pad_image_asymmetric (matrix, pad_value, kernel_shape, kernel_center):
    k_h, k_w = kernel_shape
    c_r, c_c = kernel_center

    # Amount of padding on each side
    # ekhane change hobe (kernel 180 degree flipped kintu.. must change korbi )
    pad_top = k_h - c_r - 1
    pad_left = k_w - c_c - 1
    pad_bottom = c_r
    pad_right = c_c
    # use np.pad
    pad_img = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    print(pad_img)
    
    return pad_img,(pad_top,pad_left)


def convolution_with_manual_flip(image, kernel, kernel_center_before_flip):
    """
    Perform convolution with manual 180-degree kernel flip using indexing
    """
    k_h, k_w = kernel.shape
    
    # Pad image
    padded_image, center_after_flip = pad_image_asymmetric(image, 0 ,(k_h, k_w), kernel_center_before_flip)
    
    # Output dimensions
    orig_h, orig_w = image.shape
    output = np.zeros((orig_h, orig_w))
    
    c_r, c_c = center_after_flip
    
    # Perform convolution
    for i in range(orig_h):
        for j in range(orig_w):
            acc = 0
            
            # Kernel center position in padded image"?:
            center_i = i + c_r
            center_j = j + c_c
            
            # Apply kernel with manual flip (reverse indexing)
            for m in range(k_h):
                for n in range(k_w):
                    img_i = center_i + (m - c_r)
                    img_j = center_j + (n - c_c)
                    
                    # Manual 180-degree flip: kernel[k_h-1-m][k_w-1-n]
                    img_val = padded_image[img_i][img_j]
                    kernel_val = kernel[k_h - 1 - m][k_w - 1 - n]
                    
                    acc += img_val * kernel_val
            
            output[i][j] = acc
    
    return output

# def pad_image_asymmetric(matrix, pad_value, kernel_shape, kernel_center):
#     k_h, k_w = kernel_shape
#     c_r, c_c = kernel_center

#     # Amount of padding on each side
#     # ekhane change hobe (kernel 180 degree flipped kintu.. must change korbi )
#     pad_top = c_r
#     pad_left = c_c
#     pad_bottom = k_h - c_r - 1
#     pad_right = k_w - c_c - 1


#     # use np.pad
#     pad_img = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

#     return pad_img


# def convolve2d_raw(image, kernel, kernel_center):
#     k_h, k_w = len(kernel), len(kernel[0])

#     padded_image = pad_image_asymmetric(
#         image, 0, (k_h, k_w), kernel_center
#     )

#     rows = len(padded_image)
#     cols = len(padded_image[0])

#     out_rows = rows - (k_h - 1)
#     out_cols = cols - (k_w - 1)

#     output = [[0 for _ in range(out_cols)] for _ in range(out_rows)]

#     for i in range(out_rows):
#         for j in range(out_cols):
#             acc = 0
#             for m in range(k_h):
#                 for n in range(k_w):
#                     img_r = i + m
#                     img_c = j + n
#                     # Flip indexing for convolution
#                     acc += padded_image[img_r][img_c] * \
#                         kernel[k_h - 1 - m][k_w - 1 - n]
#             output[i][j] = acc

#     return output



# Gaussian function
def gaussian_function(u, v, sigma):
    return (1 / (2 * math.pi * (sigma**2))) * math.exp(-((u**2 + v**2) / (2*sigma**2)))

# Gaussian smoothing kernel
def gaussian_smoothing_kernel(size, sigma):
    size = int(size)|1
    kernel = np.zeros((size, size))
    k = size // 2
    for i in range(-k, k+1): #x
        for j in range(-k, k+1): #y
            kernel[k-j, k+i] = gaussian_function(i, j, sigma)
    # Normalize so sum = 1
    kernel /= np.sum(kernel)
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
    #kernel -= np.mean(kernel)
    return kernel


sigma = 1.0
smoothing_kernel = gaussian_smoothing_kernel(5*sigma, sigma)
sharpening_kernel = gaussian_sharpening_kernel(7*sigma, sigma)


Gx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)
Gy = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]], dtype=np.float32)



img = np.array([[1,2,3,4,5],
              [4,5,6,6,7],
              [7,8,9,4,2],
              [17,4,9,41,2],
              [5,8,1,8,2]])

kernel = np.array([[1,2,3],[4,5,6],[7,8,9]])

c1,c2 = img.shape

output = convolution_with_manual_flip(img,kernel,(1,1))



image = cv2.imread('Box.jpg',cv2.IMREAD_GRAYSCALE)

blurred_image = convolution_with_manual_flip(image,np.flipud(np.fliplr(smoothing_kernel)), (2,2))

dx_image = convolution_with_manual_flip(image,np.flipud(np.fliplr(Gx)), (1,1))
dy_image = convolution_with_manual_flip(image,np.flipud(np.fliplr(Gy)), (1,1))

plt.figure(figsize=(8,12))
plt.imshow(blurred_image,cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(8,12))
plt.imshow(dx_image,cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(8,12))
plt.imshow(dy_image,cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()