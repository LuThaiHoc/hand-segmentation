import json
import cv2
import numpy as np
import imutils, os

#category_id 7 (left hand): (0,255,0)  
#category_id 8 (right hand): (255,0,0)  

root_data_images_file = '../Hand_data_labeled_full/'
json_file = '../Hand_data_labeled_full/final_export_json.json'
data = json.load(open(json_file))
mask_polygon_labeled_full_dir = 'mask_polygon_labeled_full/'
left_mask_dir = mask_polygon_labeled_full_dir + 'left/'
right_mask_dir = mask_polygon_labeled_full_dir + 'right/'
if not os.path.isdir(mask_polygon_labeled_full_dir):
    os.makedirs(mask_polygon_labeled_full_dir)
if not os.path.isdir(left_mask_dir):
    os.makedirs(left_mask_dir)
if not os.path.isdir(right_mask_dir):
    os.makedirs(right_mask_dir)

left_hand_color = (0,255,0)
right_hand_color = (255,0,0) 

for i, d in enumerate(data):
    
    image_name = data[d]["filename"]
    print('Image: %s\tTotal: %d / %d' % (image_name, i, len(data)))
    filename = root_data_images_file + data[d]["filename"]
    img = cv2.imread(filename)
    
    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_hand_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    right_hand_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for region in data[d]["regions"]:
        try:
            # if region["region_attributes"]["category_id"] == '7' or region["region_attributes"]["category_id"] == '8':
            if region["region_attributes"]["category_id"] == '7':
                list_points = []
                for (x,y) in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]):
                    list_points.append((x,y))
                cnt = np.array(list_points)
                cv2.drawContours(mask, [cnt], 0, left_hand_color, -1)
                cv2.drawContours(left_hand_mask, [cnt], 0, 255, -1)

            elif region["region_attributes"]["category_id"] == '8':
                list_points = []
                for (x,y) in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]):
                    cv2.circle(img, (x,y), 3, right_hand_color, 4)
                    list_points.append((x,y))
                cnt = np.array(list_points)
            
                cv2.drawContours(mask, [cnt], 0, right_hand_color, -1)
                cv2.drawContours(right_hand_mask, [cnt], 0, 255, -1)
        except Exception:
            print('Exception.')
        

    # cv2.imwrite('train_mask/' + filename, mask)
    img = imutils.resize(img, width=800)
    mask = imutils.resize(mask, width=800)
    # left_hand_mask = imutils.resize(left_hand_mask, width=800)
    # right_hand_mask = imutils.resize(right_hand_mask, width=800)
    
    cv2.imshow('mask', mask)
    cv2.imshow('img', img)
    cv2.imwrite(left_mask_dir + image_name, left_hand_mask)
    cv2.imwrite(right_mask_dir + image_name, right_hand_mask)

    # cv2.imshow('mask detect', mask_d)
    if (cv2.waitKey(1) == 27):
        break
cv2.destroyAllWindows()







'''
# print(dict_json["orig108.png2405360"])
filename = dict_json["orig108.png2405360"]["filename"]
img = cv2.imread('test/' + filename)
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for region in dict_json["orig108.png2405360"]["regions"]:
    
    list_points = []

    for (x,y) in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]):
        # print(x,y)
        cv2.circle(img, (x,y), 3, (0,0,255), 4)
        list_points.append((x,y))
 
    # print(list_points)
    cnt = np.array(list_points)
    cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)
    print(cnt)
    cv2.drawContours(img,[cnt],0,(0,255,0), 3)

    # print(cnt)
img = cv2.resize(img, (640,480))
mask = cv2.resize(mask, (640,480))
cv2.imshow('img', img)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''