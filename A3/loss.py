import numpy as np


class CrossEntropyLoss():
    """Cross entropy loss function."""

    def forward(self, Y, Y_hat):
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        
        batch_size = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat)) / batch_size

    def backward(self, Y, Y_hat):
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        
        # gradient of loss w.r.t Y_hat
        batch_size = Y.shape[0] 
        dY_hat = (-Y / Y_hat) / batch_size # element-wise division

        return dY_hat
        