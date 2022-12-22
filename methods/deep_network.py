import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy_fn, macrof1_fn

## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes, hidden_size=32):
        super(SimpleNetwork, self).__init__()

        self.conv2d = nn.Conv2d(1, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        """
        Takes as input the data x and outputs the 
        classification outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """
        x = F.relu(self.fc1(x))
        output_class = self.fc2(x)

        return output_class

class Trainer(object):

    """
        Trainer class for the deep network.
    """

    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta

        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            self.eval(dataloader_val)

            if (ep+1) % 50 == 0:
                print("Reduce Learning rate")
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.8


    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y, z = batch
            print(batch)

            # 5.2 Run forward pass.
            logits = self.model(x)
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.classification_criterion(logits, y)
            
            # 5.4 Run backward pass.
            loss.backward()
            
            # 5.5 Update the weights using optimizer.
            self.optimizer.step()
            
            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()
        

    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data. 
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            acc_run = 0
            for it, batch in enumerate(dataloader):
                x, y, z = batch
                curr_batch_size = x.shape[0]
                acc_run += accuracy_fn(self.model(x), y) * curr_batch_size
            acc = acc_run / len(dataloader.dataset)
        results_class = torch.tensor([acc])
        return results_class