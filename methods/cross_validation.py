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
    # we want to split the data into training and validation folds
    # the validation fold is the fold with index "fold"
    # the training folds are the other folds
    # the training data is the data from the training folds
    # the validation data is the data from the validation fold
    # the training labels are the labels from the training folds
    # the validation labels are the labels from the validation fold
    #
    # iterate over the folds, and append the data and labels to the training and validation
    #val_data, val_label, train_data, train_label = np.array([]), np.array([]), np.array([]), np.array([])
    D = data.shape[1]
    N = data.shape[0]
    val_data, train_data = [], []
    val_label, train_label = np.empty(fold_size), np.empty(N-fold_size)
    for i in range(len(indices)):
        if(i >= fold*fold_size and i < (fold+1)*fold_size):
            val_data.append(data[indices[i]])
            val_label = np.append(val_label, labels[indices[i]])
            
        else:
            train_data.append(data[indices[i]])
            train_label = np.append(train_label, labels[indices[i]])
        
    # reshape the data
    train_data = np.array(train_data).reshape(N-fold_size, D)
    val_data = np.array(val_data).reshape(fold_size, D)

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
            
            # split the data into training and validation folds
            train_data, train_label, val_data, val_label = splitting_fn(data, labels, indices, fold_size, fold)
            
            # fit the model on the training data
            method_obj.fit(train_data, train_label)
            
            # predict on the validation data
            preds = method_obj.predict(val_data)
            
            # compute the metric on the validation data
            metric_val = metric(val_label, preds)
            
            # append the metric to the list
            acc_list2.append(metric_val)
        # append the mean of the metric across folds to the list
        acc_list1.append(np.mean(acc_list2))

    # find the best hyper-parameter value   
    best_hyperparam = search_arg_vals[find_param_ops(acc_list1)]
    # find accuracy using the best hyper-parameter value
    best_acc = np.min(acc_list1) if method_obj.task_kind == 'regression' else np.max(acc_list1)
    
    ##
    ###
    #### YOUR CODE HERE! 
    ###
    ##

    #best_hyperparam, best_acc = 5.0, 1.0
    print("best hyperparam: ", best_hyperparam)
    print("best acc: ", best_acc)
    return best_hyperparam, best_acc

        


    