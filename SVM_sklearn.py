import numpy as np
from sklearn.svm import SVC
import pickle
import numpy as np
import cv2


class SVM_sk():
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model =  SVC(kernel='linear', probability=True)   # Initialize the SVM
        # self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', 
        #                 coef0=0.0, shrinking=True, probability=True, tol=0.001, 
        #                 cache_size=200, class_weight=None, verbose=False, max_iter=- 1, 
        #                 decision_function_shape='ovr', break_ties=False, random_state=None)

    def load(self, model_name):
        self.model = pickle.load(open(model_name, 'rb'))

    def train(self, samples, responses):
        self.model.fit(samples, responses)

    def score(self, samples, responses):
        return self.model.score(samples, responses)
    
    def predict(self, samples):
        return self.model.predict(samples)

    def predict_proba(self, samples):
        return self.model.predict_proba(samples)

    def save(self, model_name):
        pickle.dump(self.model, open(model_name, 'wb'))

    def decision_function(self, samples):
        return self.model.decision_function(samples)
      
if __name__ == "__main__":
    
    X = np.array([[2, 1,4], [6, 2,2], [5, 3,1], [3, 0,3], [5, 4,3], [1, 1,3]])
    y = np.array([0, 1, 1, 0, 1, 0])

    # classifier = SVC(kernel='linear', probability=True)   # Initialize the SVM
    # classifier.fit(X, y)                # Train the SVM

    clf = SVM_sk()
    clf.train(X, y)

    print(clf.predict_proba([[0,0,0],[7,7,7]]))
    
    print(clf.predict([[0,0,0],[7,7,7]]))

    # y = np.array([0, 1, 1, 0, 1, 1])

    # print(clf.score(X, y))