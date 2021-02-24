import ast
import numpy as np
import ast
import cv2
from utils import dis2p
import glob
# >>> x = '[ "A","B","C" , " D"]'
# >>> x = ast.literal_eval(x)

def recorvery_data_from_log(log_file):
    list_lb = []
    with open(log_file, 'r') as f:
        list_lb = f.readlines()
        f.close()

    for str_lb in list_lb:
        data = str_lb.split('-')
        image_name = data[0]
        label = ast.literal_eval(data[1])
        
        img = cv2.imread(image_name, 0)
        img = cv2.resize(img, (640,480))
        img_w = img.shape[1]
        img_h = img.shape[0]

        r_max = 0
        dis_2_center_max = 0

        for (x,y,r,lb) in label:
            if r > r_max:
                r_max = r
            if dis2p(x,y,img_w/2, img_h/2) > dis_2_center_max:
                dis_2_center_max = dis2p(x,y,img_w/2, img_h/2)

        labels = []
        for (x,y,r,lb) in label:
            dis_2_center = dis2p(x,y,img_w/2, img_h/2)
            roi = (x,y,r,r/r_max, dis_2_center,dis_2_center/dis_2_center_max)
            labels.append((roi, lb))
        
        with open('data_features/log_left_new.txt', "a") as f:
                f.write(image_name + '-' + str(labels))
                f.write('\n')
                f.close()
        

# with open('choosed_data_file_name/f_right_data_file_500.txt', 'r') as f:
#     image_paths = f.readlines()
#     f.close()

# import os
# for file in image_paths:
#     file = file.strip('\n')
#     image_name = os.path.basename(file)
#     with open('choosed_data_file_name/right_data_file_500.txt', 'a') as f:
#         f.write(image_name + '\n')
#         f.close()

# left: 1_031_1.png, 4_013_1.png, 7_077_1.png
# right: 2_036_1.png, 15_010_1.png

# names=  ['1_031_1.png', '4_013_1.png', '7_077_1.png','2_036_1.png', '15_010_1.png']
# for name in names:
#     full = 'image_demo/demo_image_envalue/' + name
#     img = cv2.imread(full)
#     img = cv2.resize(img, (640, 480))
#     cv2.imwrite('image_demo/demo_image_envalue/resized_' + name, img)

path = 'image_demo/compare/lb_03.png'
full_mask = cv2.imread(path)
full_mask = cv2.resize(full_mask, (640,480))
cv2.imwrite(path, full_mask)

# image_name = '15_010_1.png'
# full_hand_mask_path = 'data_right_hand/' + image_name
# forarm_mask_path = 'labels/right_forearm_labels/' + image_name
# hand_mask_path = 'labels/right_hand_labels/' + image_name


# full_mask = cv2.imread(full_hand_mask_path,0)
# label_forearm = cv2.imread(forarm_mask_path,0)
# label_hand = cv2.imread(hand_mask_path,0)

# demo = np.zeros((full_mask.shape[0], full_mask.shape[1], 3), dtype=np.uint8)
# demo[label_hand > 127] = (0,255,0)
# demo[label_forearm > 127] = (255,255,0)

# cv2.imwrite('image_demo/demo_image_envalue/' + image_name, full_mask)
# cv2.imwrite('image_demo/demo_image_envalue/lb_' + image_name, demo)



# demo = cv2.resize(demo, (640,480))
# full_mask = cv2.resize(full_mask, (640,480))
# label_forearm = cv2.resize(label_forearm, (640,480))
# label_hand = cv2.resize(label_hand, (640,480))

# cv2.imshow('full', full_mask)
# cv2.imshow('forearm', label_forearm)
# cv2.imshow('hand', label_hand) 
# cv2.imshow('demo', demo) 

# cv2.waitKey(0)
       

# import os
# import sklearn
# import tensorflow as tf

# print(sklearn.__version__)
# print(tf.__version__)
# filename = 'abc_1.png'
# print(os.path.splitext(filename)[0][:-2])
# img = cv2.imread('Hand_data_labeled_full/32_029.png')
# cv2.imshow('aaa', img)
# cv2.waitKey(0)

# data = np.random.randint(3, 7, (10, 1, 1, 80))
# newdata = np.squeeze(data) # Shape is now: (10, 80)
# plt.plot(newdata) # plotting by columns
# plt.show()

# r = 2.5
# a = np.array([[3,6],[2,7]])
# a = a[:, 1]

# print(a)
# a = [x/r for x in a]


# max_dis = max(a)
# if max_dis != 0:
#     a = [x / max_dis for x in a]


# print(a)

# Create a 2D Numpy array from list of lists
# arr2D = np.array([[11, 12, 13],
#                      [14, 15, 16],
#                      [17, 15, 11],
#                      [12, 14, 15]])

# m = np.argmax(arr2D, axis=None)
# print(m)

# recorvery_data_from_log('data_features/log_left.txt')

# image_names = glob.glob('mask_rcnn_output/right/*.png')
# for name in image_names:
#     img = cv2.imread(name, 0)
#     img = cv2.flip(img, 1)
#     img = cv2.resize(img, (640,480))
#     cv2.imshow('image', img)
#     cv2.waitKey(0)

"""
mask_rcnn_output/left/7_242_1.png-[((348, 293, 25.60228, 1.0, 59.941638282582836, 0.6714279993743106), 1), ((327, 329, 17.38652, 0.6791004551157163, 89.27485648266257, 1.0), 0)]
mask_data/test_left/14_086_1.png-[((53, 232, 50.59378, 1.0, 267.1198233003309, 1.0), 0), ((301, 205, 50.074387, 0.98973405, 39.824615503479755, 0.14908895570323785), 1), ((196, 230, 38.200073, 0.755035, 124.4025723206719, 0.4657182338010247), 0), ((133, 254, 18.145035, 0.3586416, 187.52333188166213, 0.7020195265359396), 0), ((373, 244, 17.38652, 0.34364936, 53.150729063673246, 0.19897710475764382), 0), ((133, 211, 17.190033, 0.33976573, 189.2353032602532, 0.7084285281496695), 0)]
"""

# a = [1,2,3]
# b = [1,2,3]

# c = np.ravel([a,b],'F').reshape(len(a), 2)
# print(c)

# a = [1,2,3,4]
# a = [x / max(a) for x in a]
# print(a)

# str_lb = 'mask_rcnn_output/left/7_242_1.png-[((348, 293, 25.60228, 1.0, 59.941638282582836, 0.6714279993743106), 1), ((327, 329, 17.38652, 0.6791004551157163, 89.27485648266257, 1.0), 0)]'
# data = str_lb.split('-')
# image_name = data[0]
# labels = ast.literal_eval(data[1])

# print(image_name)
# for roi, lb in labels:
#     print(lb, roi)
# a = (0, 1,2,3,4)

# x,y,r = a[0],a[1],a[2]
# print(x,r)

# data = np.loadtxt('data.csv', delimiter=',')
# labels = np.loadtxt('label.csv', delimiter=',')

# data = np.array(data, dtype=np.float32)
# labels = labels.astype(int)

# data = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]], dtype=np.float)
# label = np.array([1,2,3,4,5], dtype=np.int)
# # print(data.shape)
# # print(label.shape)
# # print(data)
# # print(label)

# # choice = np.random.choice(range(data.shape[0]), size=(2,), replace=False)    
# # ind = np.zeros(data.shape[0], dtype=bool)
# # ind[choice] = True
# # rest = ~ind

# s1 = np.random.choice(range(data.shape[0]), 2, replace=False)
# s2 = list(set(range(data.shape[0])) - set(s1))


# ind = np.zeros(data.shape[0], dtype=bool)
# ind[s1] = True
# rest = ~ind
# print(rest)
# # extract your samples:
# sample1 = data[s1, :]
# lb1 = label[s1]

# print(data)
# print(label)

# print(sample1)
# print(lb1)


# P = 0.9548
# F1 = 0.9688
# R = (F1*P) / (2*P - F1)
# print('Recall: %.4f' % R)