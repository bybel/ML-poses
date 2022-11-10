import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot
def logistic_regression_train_multi(self,data, labels, k, max_iters, lr):
    self.w = np.random.normal(0, 0.1, [data.shape[1], k])
    for it in range(max_iters):
        gradient = gradient_logistic_multi(data, labels, self.w)
        self.w = self.w - lr*gradient
        predictions = logistic_regression_classify_multi(data, self.w)
        if accuracy_fn(np.argmax(labels, axis=1), predictions) == 1:
            break
    return predictions
def label_to_onehot(label):
    one_hot_labels = np.zeros([label.shape[0], int(np.max(label)+1)])
    one_hot_labels[np.arange(label.shape[0]), label.astype(np.int)] = 1
    return one_hot_labels

def f_softmax(data, w):
    x=data@w
    return np.exp(x)/sum(np.exp(x))

def loss_logistic_multi(data, labels, w):
    loss = -np.sum(labels*np.log(f_softmax(data, w)))
    return loss

def gradient_logistic_multi(data, labels, w):
    return data.T@(f_softmax(data,w)-labels)

def logistic_regression_classify_multi(data, w):
    print(np.argmax(f_softmax(data,w)))
    return np.argmax(f_softmax(data,w),axis=1)

def accuracy_fn(labels_gt, labels_pred):
    """ Computes accuracy.
    
    Args:
        labels_gt (np.array): GT labels of shape (N, ).
        labels_pred (np.array): Predicted labels of shape (N, ).
        
    Returns:
        acc (float): Accuracy, in range [0, 1].
    """
    return np.sum(labels_gt == labels_pred) / labels_gt.shape[0]

class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        self.iters = 1000
        self.lr = 0.001
        if "lr" in kwargs:
            self.lr= kwargs["lr"]
        elif len(args) > 0 :
            self.lr = args[0]

        if "max_iters" in kwargs :
            self.iters = kwargs["max_iters"]
        elif len(args) > 1 :
            self.reg_arg = args[1]
       

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """
        training_labels=label_to_onehot(training_labels)
        return logistic_regression_train_multi(self,training_data, training_labels, training_labels.shape[1], self.iters, self.lr)

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   

        return logistic_regression_classify_multi(test_data, self.w)
