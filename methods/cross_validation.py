import numpy as np
from metrics import accuracy_fn, mse_fn, macrof1_fn

def splitting_fn(data, labels, indices, fold_size, fold):
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """
                
    for i in range(fold_size):
        if i == fold:
            val_data = data[indices[i*fold_size:(i+1)*fold_size]]
            val_label = labels[indices[i*fold_size:(i+1)*fold_size]]
        else:
            if i == 0:
                train_data = data[indices[i*fold_size:(i+1)*fold_size]]
                train_label = labels[indices[i*fold_size:(i+1)*fold_size]]
            else:
                train_data = np.concatenate((train_data, data[indices[i*fold_size:(i+1)*fold_size]]), axis=0)
                train_label = np.concatenate((train_label, labels[indices[i*fold_size:(i+1)*fold_size]]), axis=0)
        
    

    #train_data, train_label, val_data, val_label = None, None, None, None

    return train_data, train_label, val_data, val_label

def cross_validation(method_obj=None, search_arg_name=None, search_arg_vals=[], data=None, labels=None, k_fold=4):
    """
        Function to run cross validation on a specified method, across specified arguments.
        Arguments:
            method_obj (object): A classifier or regressor object, such as KNN. Needs to have
                the functions: set_arguments, fit, predict.
            search_arg_name (str): the argument we are trying to find the optimal value for
                for example, for DummyClassifier, this is "dummy_arg".
            search_arg_vals (list): the different argument values to try, in a list.
                example: for the "DummyClassifier", the search_arg_name is "dummy_arg"
                and the values we try could be [1,2,3]
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            k_fold (int): number of folds
        Returns:
            best_hyperparam (float): best hyper-parameter value, as found by cross-validation
            best_acc (float): best metric, reached using best_hyperparam
    """
    ## choose the metric and operation to find best params based on the metric depending upon the
    ## kind of task.
    metric = mse_fn if method_obj.task_kind == 'regression' else macrof1_fn
    find_param_ops = np.argmin if method_obj.task_kind == 'regression' else np.argmax

    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N//k_fold

    acc_list1 = []
    for arg in search_arg_vals:
        arg_dict = {search_arg_name: arg}
        # this is just a way of giving an argument 
        # (example: for DummyClassifier, this is "dummy_arg":1)
        method_obj.set_arguments(**arg_dict)

        acc_list2 = []
        for fold in range(k_fold):
            
                    
            ##
            ###
            #### YOUR CODE HERE! 
            ###
            ##
        
         
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
     
    ##
    ###
    #### YOUR CODE HERE! 
    ###
    ##

            best_hyperparam, best_acc = None, None

    return best_hyperparam, best_acc

        


    