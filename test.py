'''
import ast, os
log_file = 'data_features/right/log_right.txt'

list_lb = []
with open(log_file, 'r') as f:
    list_lb = f.readlines()
    f.close()
list_img = []
list_labels = []

i = 0
count = 0
for str_lb in list_lb:
    data = str_lb.split('-')
    image_name = data[0]
    # print(image_name)
    if not os.path.isfile(image_name):
        print('no image file!')
        count += 1
    else:
        list_img.append(image_name)
        with open('file_mask_train_svm_RIGHT.txt', 'a') as f:
            f.write(image_name + '\n')
print(list_img[0])
print(len(list_img))
print(count)
'''

import os

for name in os.listdir('labels/right_hand_labels/'):
    with open('file_mask_label_segmentation_RIGHT.txt', 'a') as f:
        f.write(name + '\n')

for name in os.listdir('labels/left_hand_labels/'):
    with open('file_mask_label_segmentation_LETF.txt', 'a') as f:
        f.write(name + '\n')