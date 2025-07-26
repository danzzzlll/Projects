from abc import abstractmethod
import numpy as np

class BaseRnnFamily:
    def __init__(self, input_dim, hidden_dim, output_dim, loss_type: str, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.loss_type = loss_type.lower()
        if self.loss_type == "mse":
            self.loss_fn = self._mse_loss_fn
            self.loss_grad_fn = self._mse_loss_grad
        elif self.loss_type == "crossentropy":
            self.loss_fn = self._crossentropy_fn
            self.loss_grad_fn = self._crossentropy_loss_grad
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _mse_loss_fn(self, y_true, y_pred):
        return np.mean(np.power(y_pred - y_true, 2))
    
    def _mse_loss_grad(self, y_true, y_pred):
        return y_pred - y_true
    
    def _crossentropy_fn(self, y_true, y_pred_logits):
        y_pred = self.softmax(y_pred_logits)
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(clipped_preds), axis=-1)
        return np.mean(loss)
    
    def _crossentropy_loss_grad(self, y_true, y_pred_logits):
        y_pred = self.softmax(y_pred_logits)
        return y_pred - y_true

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def grad(self, y_true, y_pred):
        return self.loss_grad_fn(y_true, y_pred)

    @abstractmethod
    def init_weights(self, **kwargs):
        pass

    @abstractmethod
    def forward_step(self, x_t, h_prev):
        pass

    @abstractmethod
    def backward_step(self, grad_output, x_t, h_t, h_prev):
        pass

    @abstractmethod
    def forward_sequence(self, x_seq):
        pass

    @abstractmethod
    def backward_sequence(self, x_seq, y_seq, pred_seq, debug):
        pass

    @abstractmethod
    def update_weights(self, learning_rate, optimizer, grad):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    def batchify(self, X, y, batch_size):
        for ind in range(0, len(X), batch_size):
            X_batch = X[ind: ind + batch_size, :]
            y_batch = y[ind: ind + batch_size, :]
            yield X_batch, y_batch

    def train(self, X, y, n_epochs, learning_rate, batch_size, optimizer, reset_state=None):
        for epoch_num in range(n_epochs):
            for ind, batch in enumerate(self.batchify(X, y, batch_size)):
                X_batch, y_batch = batch
                if reset_state:
                    self.reset_state()
                
                y_pred = self.forward_sequence(x_seq=X_batch)
                grad = self.backward_sequence(x_seq=X_batch, y_seq=y_batch, pred_seq=y_pred)
                self.update_weights(learning_rate=learning_rate, optimizer=optimizer, grad=grad)

    def predict(self, X, reset_state=None):
        if reset_state:
            self.reset_state()
        
        y_pred = self.forward_sequence(X)
        return y_pred
