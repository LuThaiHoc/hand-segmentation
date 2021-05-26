import cv2
import numpy as np
from utils import generate_label, vectorize_roi, get_rois, draw_rois, vectorize_rois, hand_mask_segmentation
import glob
from SVM import SVM
from SVM_sklearn import SVM_sk
import os
import time
import sklearn
import matplotlib.pyplot as plt  
from sklearn import metrics

# print(sklearn.__version__)

class PalmROIDetectionSVM:
    def __init__(self, image_size = (640,480), dimension_extract = 30, min_size_hand = 15):
        # super().__init__()
        # self.classifier = SVM()
        self.classifier = SVM_sk()

        self.image_size = image_size
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.dimension_extract = dimension_extract
        self.min_size_hand = min_size_hand
        self.seeds = [] #for random choose test samples 

        #dimension_extract must equal dimension of vector draw from center of roi to contours
        # ==> dimension of SVM vector = dimension_extract + 2  (2 added features: r_palm_normalize + dis2center_normalize)
        
    
    def load_model(self, model_name):
        self.classifier.load(model_name)

    def prepare_data_set_from_log(self, log_file, vectorization_data_file = 'data.csv', vectorization_label_file = 'label.csv'):
        import ast
        list_lb = []
        with open(log_file, 'r') as f:
            list_lb = f.readlines()
            f.close()
        list_vecs = []
        list_labels = []
        
        t = time.time()

        i = 0
        for str_lb in list_lb:
            i+=1
            print('Total : %d/%d' % (i, len(list_lb)))
            data = str_lb.split('-')
            image_name = data[0]
            # print(image_name)
            if not os.path.isfile(image_name):
                print('no image file!')
                continue
            
            img = cv2.imread(image_name, 0)
            img = cv2.resize(img, self.image_size)

            mask = img > 100
            mask = 255*mask.astype('uint8')

            labels = ast.literal_eval(data[1])

            for (roi, lb) in labels:
                # print(lb)
                # print(roi)
                # out_img = draw_rois(mask, [roi])
                # cv2.imshow('aaa', out_img)
                # cv2.waitKey(0)
                
                try:
                    vec = vectorize_roi(mask, roi, self.dimension_extract)
                except Exception:
                    continue
                list_vecs.append(vec)
                list_labels.append(lb)

        labels = np.array(list_labels)
        array_tuple = tuple(list_vecs)
        data = np.vstack(array_tuple)

        np.savetxt(vectorization_data_file, data, delimiter=',')
        np.savetxt(vectorization_label_file, labels, delimiter=',', fmt='%d')
        
        total_time = (time.time() - t)
        print('Total time: %.2f'%(total_time))
        print('FPS: %.2f'%(len(list_lb)/total_time))


    def prepare_data_set(self, data_folder, log_file_name = 'log.txt', true_label=1,
                            vectorization_data_file = 'data.csv', vectorization_label_file = 'label.csv'):
        """
            prepare data for training SVM
            dimension_extract: extrac features of roi = number of svm vector (from 10-360)
            true_label: palm left hand: 1, palm right hand: 2
            log_file_name: .txt file, save log for reprepare data train latter
            vectorization_data_file: .csv file, save data of roi vectorization
            vectorization_label_file: .csv file, save label of roivectorization
        """
        image_names = glob.glob(data_folder + '/*.*')
        print(data_folder)
        print('total image: ', len(image_names))

        # delete data in log file
        with open(log_file_name, "a") as f:
            f.truncate(0)
            f.close()

        list_vecs = []
        list_labels = []

        i = 0
        for name in image_names:
            i+= 1
            print('image name: %s, total: %d/%d' % (name, i, len(image_names)))
            img = cv2.imread(name, 0)
            
            img = cv2.resize(img, self.image_size)

            mask = img > 100
            mask = 255*mask.astype('uint8')
            
            #label[i] = (roi, label)
            #roi = (x,y,r,dis2center)

            labels = generate_label(mask, true_label, self.min_size_hand)
            
            with open(log_file_name, "a") as f:
                f.write(name.strip('\n') + '-' + str(labels))
                f.write('\n')
                f.close()
            
            for (roi, lb) in labels:
                try:
                    vec = vectorize_roi(mask, roi, self.dimension_extract)
                except Exception:
                    continue
                list_vecs.append(vec)
                list_labels.append(lb)

        labels = np.array(list_labels)
        array_tuple = tuple(list_vecs)
        data = np.vstack(array_tuple)

        np.savetxt(vectorization_data_file, data, delimiter=',')
        np.savetxt(vectorization_label_file, labels, delimiter=',', fmt='%d')
    
    def prepare_train_test_set(self, data_file, label_file, test_scale=0.3):
        data = np.loadtxt(data_file, delimiter=',')
        labels = np.loadtxt(label_file, delimiter=',')
        

        numtest = int(test_scale*data.shape[0])
        print('number of samples: ', data.shape[0])
        print('number of train: ', data.shape[0]-numtest)
        print('number of test: ', numtest)

        self.data = np.array(data, dtype=np.float32)
        self.labels = labels.astype(int)

        while True:
            seed = np.random.randint(1000000000)
            if seed not in self.seeds:
                self.seeds.append(seed)
                break
            
        np.random.seed(seed)
        choice_test = np.random.choice(range(data.shape[0]), numtest, replace=False)
        choice_train = list(set(range(data.shape[0])) - set(choice_test))

        self.train_data = self.data[choice_train, :]
        self.train_labels = self.labels[choice_train]

        self.test_data = self.data[choice_test, :]
        self.test_labels = self.labels[choice_test]
    
    def train_model(self, out_model_file, data_file, label_file, test_scale=0.3):
        self.prepare_train_test_set(data_file, label_file, test_scale)
        self.classifier.train(self.train_data, self.train_labels)
        self.classifier.save(out_model_file)
        self.calc_accuracy()
     

    def calc_accuracy(self):
        
        self.classifier_accuracy = self.classifier.score(self.test_data, self.test_labels)
        print('accuracy: %.4f' %  (self.classifier_accuracy))

        
    def inference(self, mask, alpha=1.3):
        #return: Handmask, forearm mask, palm_location
        img_out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)        
        rois = get_rois(mask, self.min_size_hand)
        vecs = vectorize_rois(mask, rois, self.dimension_extract)
        
        pre_prob = self.classifier.predict_proba(vecs)
        '''
        labels = np.argmax(pre_prob, axis=-1)
        conf = np.amax(pre_prob, axis=-1)

        # print(pre_prob)
        # print(labels)
        # print(conf)
        if np.sum(labels) == 0: #no palm detected
            return None, None, None, None
        palm_idx = -1
        max_conf = 0
        for i in range(0,len(labels)):
            if labels[i] == 1:
                if conf[i] > max_conf:
                    palm_idx = i
                    max_conf = conf[i]
        '''
        palm_idx = pre_prob.argmax(axis=0)[1]
        # print(palm_idx)
        max_conf = pre_prob[palm_idx][1]
        
        (x,y,r,r_norm, dis_2_center,dis_norm) = rois[palm_idx]
        palm_location = (x,y,r)
        hand_mask, forarm_mask = hand_mask_segmentation(mask, palm_location,alpha=alpha)
        return hand_mask, forarm_mask, palm_location, max_conf

    def calc_recall_score(self, pos_label=1):
        from sklearn.metrics import recall_score
        pre_labels = self.classifier.predict(self.test_data)
        self.classifier_recall_score = recall_score(self.test_labels, pre_labels, average='binary', pos_label=pos_label)
        print('recall score: %.4f' % (self.classifier_recall_score))
    
    def calc_auc_score(self):
        lr_probs = self.classifier.predict_proba(self.test_data)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        self.classifier_auc_score = metrics.roc_auc_score(self.test_labels, lr_probs)
        print('auc score: %.4f' % (self.classifier_auc_score))

    def calc_precision_score(self, pos_label=1):
        from sklearn.metrics import precision_score
        pre_labels = self.classifier.predict(self.test_data)
        self.classifer_precision_score = precision_score(self.test_labels, pre_labels, pos_label=pos_label)        

    # average precision score (AP)
    def calc_average_precision_score(self):
        from sklearn.metrics import average_precision_score
        y_score = self.classifier.decision_function(self.test_data)
        self.classifier_average_precision = average_precision_score(self.test_labels, y_score)
        print('average precision: %.4f' % (self.classifier_average_precision))

    def calc_F1_score(self, pos_label=1):
        from sklearn.metrics import f1_score
        pre_labels = self.classifier.predict(self.test_data)
        self.classifier_F1_score = f1_score(self.test_labels, pre_labels, pos_label=pos_label)
        print('F1 score: %.4f' % (self.classifier_F1_score))
        # print('my calc: %.4f' %( 2 * (self.calc_precision_score * self.classifier_recall_score) / (self.calc_precision_score + self.classifier_recall_score)))


    def plot_roc_curve(self, pos_label=1):
        lr_probs = self.classifier.predict_proba(self.test_data)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # print('auc score: %.4f'% metrics.roc_auc_score(self.test_labels, lr_probs))
        lr_fpr, lr_tpr, _ = metrics.roc_curve(self.test_labels, lr_probs, pos_label=pos_label)
        # print(lr_fpr, lr_tpr)
        # plot the roc curve for the model
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    def evalue_segment(self, alpha = 1.3, path = "", num_image= 200):
        path = 'labels/right_hand_labels/'
        from os import listdir
        from os.path import isfile, join
        image_names = [f for f in listdir(path) if isfile(join(path, f))]
        np.random.seed(10)
        choice = np.random.choice(len(image_names), num_image, replace=False)
        choice_img = np.array(image_names)[choice]
        # print(type(choice_img))

        precisions = []
        recalls = []
        ious = []
        dices = []

        f_precisions = []
        f_recalls = []
        f_ious = []
        f_dices = []
        
        from utils import hand_mask_segmentation_choose_max_roi
        from wrist_line import detectHandByWristCrop
        for image_name in choice_img:
            full_hand_mask_path = 'data_right_hand/' + image_name
            forarm_mask_path = 'labels/right_forearm_labels/' + image_name
            hand_mask_path = 'labels/right_hand_labels/' + image_name

            full_mask = cv2.imread(full_hand_mask_path,0)
            label_forearm = cv2.imread(forarm_mask_path,0)
            label_hand = cv2.imread(hand_mask_path,0)

            # print(full_hand_mask_path, forarm_mask_path, hand_mask_path)

            full_mask = cv2.resize(full_mask, self.image_size)
            label_forearm = cv2.resize(label_forearm, self.image_size)
            label_hand = cv2.resize(label_hand, self.image_size)

            # alpha = 1.5
            # pre_hand_mask, pre_forearm_mask, _, _ = clf.inference(full_mask, alpha=alpha)
            # pre_hand_mask, pre_forearm_mask = hand_mask_segmentation_choose_max_roi(full_mask, alpha=alpha)
            pre_hand_mask, pre_forearm_mask = detectHandByWristCrop(full_mask)

            from utils import get_iou_dice, get_pre_recall
            pre, recall = get_pre_recall(pre_hand_mask > 127, label_hand > 127)
            iou, dice = get_iou_dice(pre_hand_mask > 127, label_hand > 127)
            # pre, recall = get_pre_recall(pre_forearm_mask > 127, label_forearm > 127)
            # iou, dice = get_iou_dice(pre_forearm_mask > 127, label_forearm > 127)
            # print('hand-precision: %.4f, recal: %.4f, iou: %.4f, dice: %.4f' % (pre, recall, iou, dice))

            precisions.append(pre)
            recalls.append(recall)
            ious.append(iou)
            dices.append(dice)

            pre, recall = get_pre_recall(pre_forearm_mask > 127, label_forearm > 127)
            iou, dice = get_iou_dice(pre_forearm_mask > 127, label_forearm > 127)
            # print('fore-precision: %.4f, recal: %.4f, iou: %.4f, dice: %.4f' % (pre, recall, iou, dice))
            f_precisions.append(pre)
            f_recalls.append(recall)
            f_ious.append(iou)
            f_dices.append(dice)



            cv2.imshow('full', full_mask)
            cv2.imshow('forearm', label_forearm)
            cv2.imshow('hand', label_hand) 
            cv2.imshow('pre_hand', pre_hand_mask)
            cv2.imshow('pre_forearm', pre_forearm_mask)
            cv2.waitKey(10)
        
        cv2.destroyAllWindows()
        
        print('-------Evalue model: -------')
        print('alpha = ', alpha)
        print('hand: ')
        print('precision: %.4f'% np.array(precisions).mean())
        print('recall: %.4f'% np.array(recalls).mean())
        print('iou: %.4f'% np.array(ious).mean())
        print('dice: %.4f'% np.array(dices).mean())

        print('forearm: ')
        print('f_precision: %.4f'% np.array(f_precisions).mean())
        print('f_recall: %.4f'% np.array(f_recalls).mean())
        print('f_iou: %.4f'% np.array(f_ious).mean())
        print('f_dice: %.4f'% np.array(f_dices).mean())

    def test_images(self, dir_folder):
        img_test_name = glob.glob(dir_folder + '/*.png')
        for name in img_test_name:
            print(name)
            t = time.time()
            img = cv2.imread(name, 0)
            img = cv2.resize(img, (640,480))
            mask = img > 100
            mask = 255*mask.astype('uint8')
            hand_mask, forarm_mask, palm_location, conf = self.inference(mask)
            
            
            if hand_mask is None:
                print('no palm detected!')
                cv2.imshow('org_mask', mask)
                cv2.waitKey(0)
                continue

            to_show = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            if hand_mask is not None:
                cv2.imshow('hand_mask', hand_mask)
                to_show[hand_mask > 127] = (0,255,0)
                # print(palm_location, 'confidence: ', conf)

            if forarm_mask is not None:
                to_show[forarm_mask > 127] = (255,255,0)
                cv2.imshow('forarm_mask', forarm_mask)
            cv2.imshow('detect', to_show)
            cv2.imshow('orginal mask', img)
            # print('FPS: ', 1/(time.time()-t))
            key = cv2.waitKey(0)
            if key == 27:
                break
            if key == ord('p'):
                cv2.waitKey(0)

    def test_one_image(self, image_path, aplha = 1.3):
        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, (640,480))
        mask = img > 100
        mask = 255*mask.astype('uint8')
  
        hand_mask, forarm_mask, palm_location, conf = self.inference(mask, alpha=aplha)
        
        
        if hand_mask is None:
            print('no palm detected!')
            cv2.imshow('org_mask', mask)
            cv2.waitKey(0)
            return None

        to_show = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if hand_mask is not None:
            # cv2.imshow('hand_mask', hand_mask)
            to_show[hand_mask > 127] = (0,255,0)
            # print(palm_location, 'confidence: ', conf)

        if forarm_mask is not None:
            to_show[forarm_mask > 127] = (255,255,0)
            # cv2.imshow('forarm_mask', forarm_mask)
       
        # cv2.imshow('detect', to_show)
        # cv2.waitKey(0)
        return to_show, hand_mask, forarm_mask

def label_data(clf):
    ## Click select palm ROI to prepare dataset ######################################################
    clf.dimension_extract = 30
    features_dir =  'data_features/left/'
    di = str(clf.dimension_extract)
    # data for left hand
    clf.prepare_data_set('data_left_hand',true_label=1, log_file_name='data_features/log_left.txt',
                            vectorization_data_file=features_dir+di+'/data_left.csv',
                            vectorization_label_file=features_dir+di+'/label_left.csv')


    clf.dimension_extract = 30
    features_dir =  'data_features/left/'
    di = str(clf.dimension_extract)
    # data for right hand
    clf.prepare_data_set('data_right_hand',true_label=2, log_file_name='data_features/right/log_right.txt',
                            vectorization_data_file=features_dir+di+'/data_right.csv',
                            vectorization_label_file=features_dir+di+'/label_right.csv')
    ###################################################################################################


if __name__ == "__main__":
    clf = PalmROIDetectionSVM()
    clf.prepare_data_set('data_left_hand/', 'test.txt')
    
    
