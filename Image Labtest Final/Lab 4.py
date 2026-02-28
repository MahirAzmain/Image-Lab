

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import math

def area(binary):
    return int(np.count_nonzero(binary))

def boundary(binary):
    kernel = np.ones((3,3),dtype=np.uint8)
    eroded = cv2.erode(binary, kernel,iterations=1)
    border = binary - eroded
    return border

def perimeter(border):
    return int(np.count_nonzero(border))


def eccentricity(binary):
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours,key=cv2.contourArea)
    if(len(cnt)>=5):
        (x, y), (a, b), angle = cv2.fitEllipse(cnt)
        if a>b:
            return np.sqrt(1-(b/a)**2)
        else :
             return np.sqrt(1-(a/b)**2)

def max_diameter_bbox(binary_img):
    min_x=min_y=1e9
    max_x=max_y=0
    h,w=binary_img.shape
    for x in range(h):
        for y in range(w):
            if binary_img[x,y] > 0:
                min_x=min(min_x,x)
                min_y=min(min_y,y)
                max_x=max(max_x,x)
                max_y=max(max_y,y)
    return float(max(max_x-min_x, max_y-min_y))
         
def find_features(binary):
    A = area(binary)
    border =  boundary(binary)
    P = perimeter(border)
    maxDiameter = max_diameter_bbox(binary)
    
    compactness = P**2/A
    formfactor = (4*np.pi*A)/P**2
    roundness = (4*A)/(np.pi*maxDiameter**2)
    e = eccentricity(binary)
    
    return {
            "compactness": compactness,
            "formfactor": formfactor,
            "roundness": roundness,
            "eccentricity": e,
            }
    

train_imgs = [
    cv2.imread('c1.jpg',0),
    cv2.imread('t1.jpg', 0),
]

test_imgs = [
    cv2.imread('p3.jpg', 0),
    cv2.imread('t1.jpg', 0),
]


train_titles = ['Train 1', 'Train 2']
test_titles = ['Test 1', 'Test 2']

plt.figure(figsize=(12, 6))
plt.suptitle('Figure 1: Train and Test Image')
# --- Plot training images in the first row ---
for i, img in enumerate(train_imgs):
    plt.subplot(2, 3, i+1)   # 2 rows, 3 columns
    plt.imshow(img, cmap='gray')
    plt.title(train_titles[i])
    plt.axis('off')

# --- Plot test images in the second row ---
for i, img in enumerate(test_imgs):
    plt.subplot(2, 3, i+4)   # second row: positions 4,5,6
    plt.imshow(img, cmap='gray')
    plt.title(test_titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('1.png',dpi=300)
plt.show()

keys = ["compactness", "formfactor", "eccentricity","roundness"]

train_vec = []
for im in train_imgs:
    desc = find_features(im)
    vec = []
    for k in keys:
        vec.append(desc[k])
    
    train_vec.append(vec)

test_vec = []
for im in test_imgs:
    desc = find_features(im)
    vec = []
    for k in keys:
        vec.append(desc[k])
    test_vec.append(vec)
    
def eucledian_distance(train_vec,test_vec):
    temp = 0;
    for i in range(len(train_vec)):
        temp += (train_vec[i]-test_vec[i])**2 
    
    return np.sqrt(temp.astype(np.float32))

def cosine_similarity(v1, v2):
    dot = 0.0
    for i in range(len(v1)):
        dot += v1[i] * v2[i]

    n1 = 0.0
    for i in range(len(v1)):
        n1 += v1[i] * v1[i]
    n1 = np.sqrt(n1)

    n2 = 0.0
    for i in range(len(v2)):
        n2 += v2[i] * v2[i]
    n2 = np.sqrt(n2)

    if n1 == 0 or n2 == 0:
        return 0.0
    else:
        return dot / (n1 * n2)

def kullback_leibler(P, Q):
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    dist = 0.0
    eps = 1e-10  # to avoid log(0)
    for i in range(P.shape[0]):
        dist += P[i] * math.log((P[i] + eps) / (Q[i] + eps), 2)
    return dist

distance_mat = []
for tvec in test_vec:
    row = []
    for trvec in train_vec:
        dist = kullback_leibler(tvec, trvec)
        row.append(dist)
    distance_mat.append(row)
    
row_headers = [f'Test {i + 1}' for i in range(len(test_vec))]
col_headers = [f'Train {i + 1}' for i in range(len(train_vec))]

print(tabulate(distance_mat, headers=col_headers, showindex=row_headers, tablefmt='grid')) 