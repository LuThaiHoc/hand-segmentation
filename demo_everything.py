from palm_roi_detection_SVM import *
from utils import *

def demo_handsegment_method():
    full_mask = cv2.imread('image_demo/compare/02.png',0)
    full_mask = cv2.resize(full_mask, (640,480))
    from utils import hand_mask_segmentation_choose_max_roi
    from wrist_line import detectHandByWristCrop
    # hand_mask, forearm_mask = hand_mask_segmentation_choose_max_roi(full_mask, 1.3)
    hand_mask, forearm_mask = detectHandByWristCrop(full_mask)
    

    to_show = np.zeros((full_mask.shape[0], full_mask.shape[1], 3), dtype=np.uint8)
    to_show[hand_mask > 127] = (0,255,0)
    to_show[forearm_mask > 127] = (255,255,0)
    cv2.imshow('res', to_show)
    cv2.waitKey(0)


def demo_images(img_name):
    clf = PalmROIDetectionSVM(image_size = (640,480), dimension_extract = 30, min_size_hand = 15)
    clf.load_model('models/left_30.pkl')
    clf_r = PalmROIDetectionSVM(image_size = (640,480), dimension_extract = 30, min_size_hand = 15)
    clf_r.load_model('models/right_30.pkl')
    
    # img_name = '4_024.png'
    org_image = 'bad_case_demo/org/' + img_name
    left_mask = 'bad_case_demo/mask/left/' + img_name
    right_mask = 'bad_case_demo/mask/right/' + img_name
    
    org = cv2.imread(org_image)
    org = cv2.resize(org, (640,480))

    res = None
    res_r = None

    if os.path.isfile(left_mask):
        res = clf.test_one_image(left_mask)
        cv2.imshow('res', res)
        # org = cv2.addWeighted(org, 0.7, res, 0.3, 0)

    if os.path.isfile(right_mask):
        res_r = clf_r.test_one_image(right_mask)
        cv2.imshow('res_r', res_r)
        # org = cv2.addWeighted(org, 0.7, res_r, 0.3, 0)
    if res_r is not None and res is not None:
        add = cv2.addWeighted(res, 1, res_r, 1, 1)
    elif res_r is not None:
        add = res_r
    else:
        add = res

    final = cv2.addWeighted(org, 0.7, add, 0.3, 1)
    from utils import apply_brightness_contrast
    final = apply_brightness_contrast(final, brightness=50, contrast=40)
    cv2.imshow('add', final)
    cv2.waitKey(0)
    cv2.imwrite('bad_case_demo/final/' + img_name, final)


def evaluate_segment(clf):
    clf.test_images('image_demo/compare/')
    # left: 1_031_1.png, 4_013_1.png, 7_077_1.png
    # right: 2_036_1.png, 15_010_1.png

    names=  ['2_036_1.png', '15_010_1.png']
    for name in names:
        full = 'image_demo/demo_image_envalue/' + name
        # image_name = '15_010_1.png'
        # full_hand_mask_path = 'data_right_hand/' + image_name
        clf.load_model('models/right_30.pkl')
        res = clf.test_one_image(full, aplha=1.7)
        cv2.imshow('res', res)
        cv2.imwrite('image_demo/demo_image_envalue/1.7/' + name, res)
        cv2.waitKey(0)
    
    clf.evalue_segment(alpha=1.1)    
    clf.evalue_segment(alpha=1.5)

    for i in range(8):
        print('envalue with alpha = %f' % (1+i/10))
        clf.evalue_segment(alpha=1+i/10)

    clf.evalue_segment()


def average_train(clf):
    ## average train
    # clf.dimension_extract = 90
    save_model_name = 'left_temp.pkl'
    features_dir =  'data_features/right/'
    di = str(clf.dimension_extract)
    times_train = 30
    
    accuracy = []
    precisions = []
    recalls = []
    F1_scores = []

    for i in range(times_train):
        clf.train_model(save_model_name, features_dir +di+'/data_right.csv', features_dir+di+'/label_right.csv', test_scale=0.3)
        clf.calc_accuracy()        
        clf.calc_recall_score(pos_label=2)
        clf.calc_precision_score(pos_label=2)
        clf.calc_F1_score(pos_label=2)
        accuracy.append(clf.classifier_accuracy)
        precisions.append(clf.classifer_precision_score)
        recalls.append(clf.classifier_recall_score)
        F1_scores.append(clf.classifier_F1_score)
        print('-------')
        
    
    print('----------------')
    print('dimension extract: ', clf.dimension_extract)
    print('avg accuracy: %.4f'%(np.array(accuracy).mean()))
    print('avg recall: %.4f'%(np.array(recalls).mean()))
    print('avg precision: %.4f'%(np.array(precisions).mean()))
    print('avg F1 score: %.4f'%(np.array(F1_scores).mean()))
    # print('avg recall score: %.4f'%(rc/times_train))
    

def train_and_evalue_model(clf):
    # clf.dimension_extract = 30
    # save_model_name = 'models/left_30.pkl'
    save_model_name = 'test.pkl'

    features_dir =  'data_features/left/'
    di = str(clf.dimension_extract)
    # clf.prepare_data_set_from_log('data_features/left/log_left.txt', 
    #                                 vectorization_data_file=features_dir+di+'/data_left.csv',
    #                                 vectorization_label_file=features_dir+di+'/label_left.csv')
    clf.train_model(save_model_name, features_dir +di+'/data_left.csv', features_dir+di+'/label_left.csv', test_scale=0.3)
    clf.prepare_train_test_set(features_dir +di+'/data_left.csv', features_dir+di+'/label_left.csv', test_scale=0.3)
    clf.load_model(model_name=save_model_name)
    clf.calc_recall_score()
    clf.calc_auc_score()
    clf.calc_precision_score()
    clf.calc_F1_score()
    print(clf.classifier_recall_score)
    clf.test_images('data_left_hand')
    # clf.plot_roc_curve(pos_label=1)
    
    ### RIGHT
    # clf.dimension_extract = 30
    # save_model_name = 'models/right_30.pkl'
    # features_dir =  'data_features/right/'
    # di = str(clf.dimension_extract)
    # # clf.prepare_data_set_from_log('data_features/right/log_right.txt', 
    # #                                 vectorization_data_file=features_dir+di+'/data_right.csv',
    # #                                 vectorization_label_file=features_dir+di+'/label_right.csv')
    # clf.train_model(save_model_name, features_dir +di+'/data_right.csv', features_dir+di+'/label_right.csv', test_scale=0.3)
    
    # clf.load_model(model_name=save_model_name)
    # # clf.test_images('data_right_hand')
    # clf.plot_roc_curve(pos_label=2)

    ####################################################################################################

if __name__ == "__main__":
    clf = PalmROIDetectionSVM(image_size = (640,480), dimension_extract = 30, min_size_hand = 15)
    clf.dimension_extract = 30
    clf.load_model('models/left_30.pkl')

    # for img_name in os.listdir('bad_case_demo/org/'):
    #     demo_images(img_name)

    # evaluate_segment(clf)

    # train_and_evalue_model(clf)

    # demo_handsegment_method()