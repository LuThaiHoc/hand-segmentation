import cv2
import numpy as np
import math
from scipy.spatial import distance
import glob

def dis2p(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def closest_point(point, points):
    if len(points) == 0:
        return (320,240)
    closest_index = distance.cdist([point], points).argmin()
    # print(closest_index)
    return points[closest_index]

def vectorize_roi(mask, roi, vec_dimension): 
    assert vec_dimension <= 360, 'vec_dimension must be < 360, min step is 1 degree'

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = mask.copy()

    ref = np.zeros_like(mask)
    cv2.drawContours(ref, contours, 0, 255, 1)

    (centroid_x,centroid_y,radius,radius_nor, dis_2_center,dis_2_center_nor) = roi

    # Get dimensions of the image
    width = mask.shape[1]
    height = mask.shape[0]

    # cv2.imshow('ref', ref)
    
    vec = []
    for i in range(vec_dimension):

        tmp = np.zeros_like(mask)

        theta = i*(360/vec_dimension)
        theta *= np.pi/180.0

        cv2.line(tmp, (centroid_x, centroid_y),
                 (int(centroid_x+np.cos(theta)*width),
                  int(centroid_y-np.sin(theta)*height)), 255, 5)

        (row, col) = np.nonzero(np.logical_and(tmp, ref))
        # print(row, col)
        closest_p = closest_point((centroid_x, centroid_y),np.ravel([col,row],'F').reshape(len(row), 2))

        cv2.line(out, (centroid_x, centroid_y), (closest_p[0], closest_p[1]), 0, 1)
        
        # vec.append(dis2p(centroid_x, centroid_y, closest_p[0], closest_p[1])/radius)
        vec.append(dis2p(centroid_x, centroid_y, closest_p[0], closest_p[1]))

    # cv2.imshow('out', out)
    # cv2.waitKey(0)

    max_dis = max(vec)
    if max_dis != 0:
        vec = [x / max_dis for x in vec]
    # add radius scale in image as one feature of hand palm
    vec.append(radius_nor)
    # add dis2center scale in image as one feature of hand palm
    vec.append(dis_2_center_nor)
    vec = np.array(vec, dtype=np.float32)

    # print('vec: ', vec)
    return vec

def vectorize_rois(mask, rois, vec_dimension):
    vecs = []
    for roi in rois:
        vecs.append(vectorize_roi(mask, roi, vec_dimension))
    vecs = np.array(vecs)
    return vecs


def getDistanceTransform(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    r = np.amax(dist)  
    indices = np.where(dist == r)
    y,x = indices[0][0], indices[1][0]
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    return (x,y), r, dist

#return rois (region can be location of palm)
def get_rois(mask, thresh):
    maskcp = np.copy(mask)
    img_w = mask.shape[1]
    img_h = mask.shape[0]
    rois = []
    r_max = 0
    dis_2_center_max = 0
    while True:
        dist = cv2.distanceTransform(maskcp, cv2.DIST_L2, 3)
        r = np.amax(dist)  
        if r >= thresh: #only accept roi with radius > thresh
            indices = np.where(dist == r)
            y,x = indices[0][0], indices[1][0]
            dis_2_center = dis2p(x,y,img_w/2, img_h/2)
            
            rois.append((x,y,r, dis_2_center))
            #hide this added roi
            cv2.circle(maskcp, (x,y), int(1.3*r), (0,0,0),-1)

            if r > r_max:
                r_max = r
            if dis_2_center > dis_2_center_max:
                dis_2_center_max = dis_2_center
        else:
            break
    
    roi_final = []
    for roi in rois:
        (x,y,r, dis_2_center) = roi
        roi_final.append((x,y,r,r/r_max, dis_2_center,dis_2_center/dis_2_center_max))

    #each roi contains: x,y,r,(normalize of r to 0-1), dis2center, (normalize of dis2center to 0-1)
    return roi_final

def draw_rois(mask, rois):
    img_show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    img_show[mask == 255] = (255,255,255)
    for roi in rois:
        x,y,r = roi[0], roi[1], roi[2]
        cv2.circle(img_show, (x,y), int(1.0*r), (0,0,255), 2)
    return img_show


point = (0, 0)
def capture_event(event, x, y, flags, params):
    global point
    # Check if the event was left click
    if event == cv2.EVENT_LBUTTONDOWN:
        pre_point = point
        point = (x, y)
        print('clicked: ', (x, y))

def generate_label(mask, true_label, min_size_hand = 20):
    window_name = 'select roi which is palm center'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000,800)
    # cv2.resizeWindow(window_name, 640,480)
    cv2.moveWindow(window_name, 400, 200)
    # set the mouse settin function
    cv2.setMouseCallback(window_name, capture_event)

    
    rois = get_rois(mask, thresh=min_size_hand)
    # mask_cp = np.copy(mask)

    while True:
        img_show = draw_rois(mask, rois)
        cv2.circle(img_show, point, 3, (0,255,0), 4)
        cv2.imshow(window_name, img_show)
        key = cv2.waitKey(10)
        if key == 13 or key == ord('q'):
            cv2.destroyAllWindows()
            break

    print('select point: ', point)
    labels = []
    for roi in rois:
        # each roi contains: x,y,r,(normalize value of r to [0-1]), dis2center, (normalize value of dis2center to [0-1])
        x,y,r = roi[0], roi[1], roi[2]
        if dis2p(x,y, point[0], point[1]) < r:
            labels.append((roi,true_label))
        else:
            labels.append((roi,0))
    return labels


def choose_image_train(out_put_file, root_folder):
    import glob
    names = glob.glob(root_folder + '/*.png')
    total = len(names)
    i = 0
    choosed = 0
    for name in names:
        i+=1
        print(name, i, '/', total, '-------------', choosed)
        
        img = cv2.imread(name, 0)
        img = cv2.resize(img, (640,480))
        mask = img > 128
        mask = img > 128
        mask = 255*mask.astype('uint8')
        rois = get_rois(mask, 15)
        out = draw_rois(mask, rois)
        cv2.imshow('mask', out)
        key = cv2.waitKey(0)
        if key == 13:
            choosed += 1
            with open(out_put_file, 'a') as f:
                f.write(name + '\n')
                f.close()
                
def findContours(mask):
    if cv2.getVersionMajor() in [2, 4]:
        # OpenCV 2, OpenCV 4 case
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_NONE)
    else:
        # OpenCV 3 case
        _, cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)
    return cnts     

#input image 
#return: mask of hand and mask of forearm
def hand_mask_segmentation(mask, palm_location, alpha = 1.3,hand='left'):
    assert hand == 'left' or hand == 'right', 'hand must be left or right'    

    x_palm, y_palm, r_palm = palm_location
    # print(x_palm, y_palm, r_palm)

    mask_cp = np.copy(mask)
    mask_cp = cv2.circle(mask_cp, (x_palm,y_palm), int(alpha*r_palm), (0,0,0),-1)
    

    width = mask.shape[1]
    height = mask.shape[0]
    # if hand == 'left':
    #     mask_cp[:, x_palm:width] = np.zeros((height, width-x_palm), dtype=np.uint8)
    # elif hand == 'right':
    #     mask_cp[:, :x_palm] = np.zeros((height, x_palm), dtype=np.uint8)

    dist = cv2.distanceTransform(mask_cp, cv2.DIST_L2, 3)
    r = np.amax(dist) 
    indices = np.where(dist == r)
    y,x = indices[0][0], indices[1][0]


    if (r > r_palm/3.0): #forearm
        cnts = findContours(mask_cp)
        for cnt in cnts:
            d = cv2.pointPolygonTest(cnt, (x,y), True) #check if (X,Y) inside the contour
            if (d < 0): #is not the forearm
                cv2.drawContours(mask_cp, [cnt], 0, (0,0,0), -1)
    else:
        # return only hand, no forarm
        return mask, np.zeros(mask.shape, dtype=np.uint8)

    hand_mask = np.copy(mask)
    hand_mask[mask_cp > 0] = 0 

    #hand mask is mask of hand, maskcp is mask of forearm(mask after remove all contour not in forearm)
    return hand_mask, mask_cp 


#input image 
#return: mask of hand and mask of forearm
def hand_mask_segmentation_choose_max_roi(mask, alpha= 1.3, hand='left'):
    assert hand == 'left' or hand == 'right', 'hand must be left or right'    

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    r_palm = np.amax(dist) 
    indices = np.where(dist == r_palm)
    y_palm,x_palm = indices[0][0], indices[1][0]
    # print(x_palm, y_palm, r_palm)

    mask_cp = np.copy(mask)
    mask_cp = cv2.circle(mask_cp, (x_palm,y_palm), int(alpha*r_palm), (0,0,0),-1)
    

    width = mask.shape[1]
    height = mask.shape[0]
    # if hand == 'left':
    #     mask_cp[:, x_palm:width] = np.zeros((height, width-x_palm), dtype=np.uint8)
    # elif hand == 'right':
    #     mask_cp[:, :x_palm] = np.zeros((height, x_palm), dtype=np.uint8)

    dist = cv2.distanceTransform(mask_cp, cv2.DIST_L2, 3)
    r = np.amax(dist) 
    indices = np.where(dist == r)
    y,x = indices[0][0], indices[1][0]


    if (r > r_palm/3.0): #forearm
        cnts = findContours(mask_cp)
        for cnt in cnts:
            d = cv2.pointPolygonTest(cnt, (x,y), True) #check if (X,Y) inside the contour
            if (d < 0): #is not the forearm
                cv2.drawContours(mask_cp, [cnt], 0, (0,0,0), -1)
    else:
        # return only hand, no forarm
        return mask, np.zeros(mask.shape, dtype=np.uint8)

    hand_mask = np.copy(mask)
    hand_mask[mask_cp > 0] = 0 

    #hand mask is mask of hand, maskcp is mask of forearm(mask after remove all contour not in forearm)
    return hand_mask, mask_cp 

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def get_pre_recall(outputs: np.array, labels: np.array, smooth=1e-6):
    """Calculate precision, recall

    @param outputs: Predict value
    @type  outputs: numpy.array

    @param labels: Label value
    @type  labels: numpy.array

    @return: precision, recall
    @rtype : float, float
    """
    # outputs = outputs.squeeze(1)

    true = np.sum(labels)    # true = TP + FN
    pred = np.sum(outputs)   # pred = TP + FP

    intersection = np.sum(outputs * labels)    # intersection = TP
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    return (intersection + smooth) / (pred + smooth), (intersection + smooth) / (true + smooth)


def get_iou_dice(outputs: np.array, labels: np.array, smooth=1e-6):
    """Calculate iou, dice

    @param outputs: Predict value
    @type  outputs: numpy.array

    @param labels: Label value
    @type  labels: numpy.array

    @return: iou, dice
    @rtype : float, float
    """
    true = np.sum(labels)    # true = TP + FN
    pred = np.sum(outputs)   # pred = TP + FP

    intersection = np.sum(outputs * labels)    # intersection = TP
    return (intersection + smooth) / (true + pred - intersection + smooth), (2*intersection + smooth) / (true + pred + smooth)


if __name__ == "__main__":
    # with open('right_data_file_500.txt', 'r') as f:
    #     choose_images = f.readlines()
    #     f.close()
    # import shutil
    # import os
    # # for img_files in choose_images:
    # #     shutil.copy(img_files.strip('\n'), 'data_right_hand/' + os.path.basename(img_files))
    # images = glob.glob('data_right_hand/*.*')
    # for name in images:
    #     os.rename(name, os.path.splitext(name)[0] + '.png')

    # choose_image_train('right_data_file_500.txt', 'mask_rcnn_handata_output_only_mask/right/')
    # img = cv2.imread('img.jpg', 0)
    # # img = cv2.resize(img, (640,480))
    # mask = img > 100
    # mask = 255*mask.astype('uint8')
    # cv2.imshow('mask', mask)

    # hand_mask, mask_cp = hand_mask_segmentation(mask, 1)
    # cv2.imshow('hand_mask', hand_mask)
    # cv2.imshow('fore_arm', mask_cp)
    # cv2.waitKey(0)
    
    img = cv2.imread('image_demo/new_img_demo/01/demo.png', 0)
    img = cv2.resize(img, (640,480))
    cv2.imwrite('image_demo/new_img_demo/01/demo.png', img)
    mask = img > 100
    mask = 255*mask.astype('uint8') 
    
    (x,y), r, dist = getDistanceTransform(mask)

    rois = get_rois(mask, 20)
    out_img = draw_rois(mask, [rois[0]])
    cv2.imshow('out', out_img)
    cv2.imshow('outdist', dist)
    cv2.waitKey(0)
    
    # for i in range(0, len(rois)):
    #     (x,y,r,r_norm, dis_2_center,dis_2_center_norm) = rois[i]
    #     cv2.circle(out_img, (x,y), 1, (0,255,0), 2)
    #     cv2.putText(out_img,'O' , (x+3,y-3), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0))
    #     cv2.putText(out_img,str(i+1), (x+25,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))
    

    # dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
    # (x,y,r,r_norm, dis_2_center,dis_2_center_norm) = rois[0]
    # cv2.circle(dist, (x,y), 2, (0,255,0), 3)
    # cv2.putText(dist,'O' , (x+3,y-3), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
    # cv2.putText(dist,str(i+1), (x+25,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))

    # cv2.imshow('out', out_img)
    # cv2.imshow('mask', dist)
    # cv2.imshow('demo', img)
    # cv2.waitKey(0)

    
    
    
    
    
    
    
    
    '''
    # get new label file
    with open('left_data_file_500.txt', 'r') as f:
        name_choose = f.readlines()
        
    x = 'aa'
    x.strip('\n')
    name_choose = [x.strip('\n') for x in name_choose]
    
    with open('data_features/log_left.txt', 'r') as f2:
        old = f2.readlines()
    
    print('total image: ', len(name_choose))
    tt = 0
    for lb in old:
        data = lb.split('-')
        if data[0] in name_choose:
            with open('data_features/log_left_500.txt', 'a') as f:
                f.write(lb)
                tt += 1
    print('ttt', tt)
    '''


    
