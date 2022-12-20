import numpy as np
import metrics

class KNN(object):
    """
        kNN classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """
    train_data = None
    train_labels = None

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
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        self.k = 6
        if "k" in kwargs:
            self.k= kwargs["k"]
        elif len(args) > 0 :
            self.k = args[0]
        


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.train_data = training_data
        self.train_labels = training_labels
        pred_labels = training_labels
        
        return pred_labels
    
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """      
        test_labels = np.zeros(len(test_data))
        for i in range(len(test_data)):
            distances = metrics.euclidean_distance(test_data[i], self.train_data)
            indices = find_k_nearest_neighbors(self.k, distances)
            labels = self.train_labels[indices]
            test_labels[i] = np.bincount(labels).argmax()
        return test_labels
    
    
    
def find_k_nearest_neighbors(k, distances):
    """ Find the indices of the k smallest distances from a list of distances.
        Tip: use np.argsort()
    """
    indices = np.argsort(distances)
    
    return indices[:k]