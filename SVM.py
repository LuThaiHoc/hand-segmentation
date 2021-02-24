import numpy as np
import cv2

#bad model, let using SVM_sklearn.py
class SVM():
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        # Set up SVM for OpenCV 3
        self.model = cv2.ml.SVM_create()
        # Set SVM type
        self.model.setType(cv2.ml.SVM_C_SVC)
        # Set SVM Kernel to Radial Basis Function (RBF) 
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        # svm.setKernel(cv2.ml.SVM_RBF)

        self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        # Set parameter C
        self.model.setC(1)
        # Set parameter Gamma
        self.model.setGamma(0.50625)



    def load(self, model_name):
        self.model = cv2.ml.SVM_load(model_name)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        # value of predict like that
        # (0.0, array([[0.],
        #             [0.],
        #             [0.],
        #             [1.]], dtype=float32))
        return self.model.predict(samples)[1].reshape(len(samples))

    def score(self, samples, responses):
        total_test = len(samples)
        true_case = 0
        
        for (sample, label) in zip(samples, responses):
            sample = sample.reshape(1, len(sample))
            pre = self.model.predict(sample)
            if label == int(pre[0]):
                true_case += 1
        return (true_case/total_test)

    def save(self, model_name):
        self.model.save(model_name)
      

if __name__ == "__main__":
    samples = np.array(np.random.random((4,2)), dtype = np.float32)
    labels = np.array([1.,0.,0.,1.], dtype = np.int)

    # print(samples)
    print(labels)

    clf = SVM()
    clf.train(samples, labels)
    # samples = np.reshape(4,2)

    y_val = clf.predict(samples)
    print(y_val)