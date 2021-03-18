from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import argparse
from scipy.signal import find_peaks
from math import atan2, cos, sin, sqrt, pi
import imutils
import glob


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# reference: https://docs.opencv2.org/3.4/d1/dee/tutorial_introduction_to_pca.html
def getRotateAngle(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1] sss
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(cntr[0]), int(cntr[1])), (0,120,255), 2, cv2.LINE_AA)
    
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    cv2.line(img, (int(p2[0]), int(p2[1])), (int(cntr[0]), int(cntr[1])), (120,0,255), 2, cv2.LINE_AA)
    

    cv2.circle(img, (int(img.shape[1]/2), int(img.shape[0]/2)), 3, (50,100,125), 2)
    
    cv2.imwrite('for_show/PAC.jpg', img)

    # cv2.imshow('pre', img)
    alpha = atan2(p1[0] - cntr[0] , p1[1] - cntr[1]) #radian value

    # print('eigenvector = ', eigenvectors[0,1])

    #return degree angle
    if (eigenvectors[0,1] > 0):
        return -alpha * 180 / pi 
    return 180 -alpha * 180 / pi
    # return -alpha * 180 / pi #convert radian to degree

r_angle = 0
#input: gray image
def Wrist_cropping(gray):
    #smoothing image
    
    process_img = np.copy(gray)
    #to draw every thing
    process_img = cv2.cvtColor(process_img, cv2.COLOR_GRAY2BGR)
    rotated = np.zeros(gray.shape, dtype=np.uint8)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    # Convert image to binary
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if cv2.getVersionMajor() in [2, 4]:
        # OpenCV 2, OpenCV 4 case
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    else:
        # OpenCV 3 case
        _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours, _ = cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) 

    angle = 0
    if (len(contours) != 0):
        maxCnt = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(maxCnt)
        cv2.drawContours(rotated, [maxCnt], 0, (255,255,255), -1)
        angle = getRotateAngle(maxCnt, process_img)
    else:
        return process_img, -1, rotated, angle
    
    # cv2.imshow('rotated', rotated)
    # r_angle = angle

    rows = rotated.shape[0]
    cols = rotated.shape[1]
    # rotatedN = rotate_image_new(rotated, angle)
    # cv2.imshow('NNNN',rotatedN)
    temp = np.copy(rotated)
    temp = rotate_image(temp, angle)
    if np.count_nonzero(temp[rows-30, :]) < 10:
        # rotated = rotate_image(rotated, 180)
        angle = 180+angle

    rotated = rotate_image(rotated, angle)
    process_img = rotate_image(process_img, angle)
    

    rotated = cv2.erode(rotated, np.ones((5,5), np.uint8), iterations=1) 
    # rotated = cv2.erode(rotated, np.ones((15,15), np.uint8), iterations=1) 
    # rotated = cv2.dilate(rotated, np.ones((5,5), np.uint8), iterations=1)
    # print(rows)

    dist = np.zeros(rows)


    for i in range(0, rows-1):
        indexs = np.where(rotated[i,:] == 255)
        if (len(indexs[0]) > 0):
            s = indexs[0][0]
            e = indexs[0][len(indexs[0])-1]
            # print(indexs)
            dist[i] = e-s
            cv2.circle(process_img, (s, i), 1, (0, 0, 255), 1)
            cv2.circle(process_img, (e, i), 1, (0, 255, 0), 1)

    peaks, _ = find_peaks(-dist, distance=30)  #Indices of peaks in x that satisfy all given conditions.
    num_peaks = len(peaks)

    # print(peaks)
    # print(dist)

    for i in range(0, num_peaks):
        cv2.line(process_img, (0, peaks[i]), (process_img.shape[1], peaks[i]), (255,255,0), 1, cv2.LINE_AA)

    cv2.imwrite('peak.jpg', process_img)
    if (num_peaks == 0): #cannot detect
        return process_img, -1, rotated, rows
    elif (num_peaks <= 2): #choose the first peak
        cv2.line(process_img, (0, peaks[num_peaks-1]), (process_img.shape[1], peaks[num_peaks-1]), (0,255,255), 2, cv2.LINE_AA)
        return process_img, peaks[num_peaks-1], rotated, angle
    

    # print(peaks)
    distances = np.zeros(num_peaks - 1)
    for i in range(0, num_peaks-1):
        distances[i] = peaks[i+1] - peaks[i]

    # print(distances) 
   
    avg = distances.mean()
    temp = np.square(np.subtract(distances, avg))
    variance = np.sum(temp) / len(temp)

    # print('avg= ', avg, ' variance= ', sqrt(variance))
    th = avg + 0.7*sqrt(variance)
    # print('thresh= ', th)

    peak = peaks[num_peaks - 1]
    for i in range(num_peaks - 2, -1, -1):
        if (peak - peaks[i] > th):
            break
        else:
            peak = peaks[i]
    # peak = peaks[2]
    # print('choose peak = ', peak)
    cv2.line(process_img, (0, peak), (process_img.shape[1], peak), (0,255,255), 2, cv2.LINE_AA)
    return process_img, peak, rotated, angle

def detectHandByWristCrop(image):
    # image = cv2.imread(path, 0)
    detect = np.copy(image)
    
    # gray = get_skin_mask(image)
    # cv2.imshow('maskgray', gray)
    # getMask()
    

    process_img, loc, rotate_mask, rotate_angle = Wrist_cropping(image)

    hand_mask = np.copy(rotate_mask)
    forearm_mask = np.copy(rotate_mask)


    hand_mask[loc: hand_mask.shape[1], :] = 0
    forearm_mask[0: loc, :] = 0
    hand_mask = rotate_image(hand_mask, -rotate_angle)
    forearm_mask = rotate_image(forearm_mask, -rotate_angle)


    # cv2.imshow('hand', hand_mask)
    # cv2.imshow('forearm', forearm_mask)
    # cv2.imshow('process img', process_img)
    # cv2.imshow('rotate mask', rotate_mask)
    # cv2.waitKey(0)
    return hand_mask, forearm_mask

if __name__ == "__main__":
    import os, glob
    files = glob.glob('data_left_hand/*.png')
    for f in files:
        img = cv2.imread(f, 0)
        img = cv2.resize(img, (640,480))
        
        h, f = detectHandByWristCrop(img)
        cv2.imshow('hand', h)
        cv2.imshow('forearm', f)
        cv2.waitKey(0)