import cv2
from utils import *
import glob, os
from tqdm import tqdm
from palm_roi_detection_SVM import PalmROIDetectionSVM
import shutil

r_dir = 'predict_test/right/'
if not os.path.isdir(r_dir):
    os.makedirs(r_dir)

f_mask = r_dir + 'right_full/'
f_hand = r_dir + 'right_hand/'
f_forearm = r_dir + 'right_forearm/'

print(f_mask, f_hand, f_forearm)

if not os.path.isdir(f_mask):
    os.makedirs(f_mask)
if not os.path.isdir(f_hand):
    os.makedirs(f_hand)
if not os.path.isdir(f_forearm):
    os.makedirs(f_forearm)


image_name = glob.glob('labels/right_hand_labels/*.png')
image_name = [f[:-6] + '.png' for f in image_name]
res = []
res  = [res.append(x) for x in image_name if x not in res]
print(len(res))
print(len(image_name))

image_name = [os.path.basename(f) for f in image_name]


clf = PalmROIDetectionSVM(image_size = (640,480), dimension_extract = 30, min_size_hand = 15)
clf.dimension_extract = 30
clf.load_model('models/right_30.pkl')

for f in tqdm(image_name):
    path = 'mask_rcnn_handata_output_only_mask/right/' + f[:-4] + '_1.png'

    if not os.path.isfile(path):
        continue
    
    to_show, hand_mask, forarm_mask = clf.test_one_image(path, aplha=1.5)

    full_mask = cv2.imread(path,0)
    full_mask = cv2.resize(full_mask, (640,480))
    
    cv2.imwrite(f_mask + f, full_mask)
    cv2.imwrite(f_hand + f, hand_mask)
    cv2.imwrite(f_forearm + f, forarm_mask)
    
    # cv2.imshow('toshow', to_show)
    # cv2.waitKey(0)