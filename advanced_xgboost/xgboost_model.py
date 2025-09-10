import numpy as np
from .tree import Tree
from typing import Tuple

class XGBoost:
    """
    A senior-level, from-scratch implementation of XGBoost for regression.
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_child_weight: int = 1,
                 lambda_reg: float = 1.0, gamma_reg: float = 0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_reg = lambda_reg  # L2 regularization term
        self.gamma_reg = gamma_reg    # Minimum gain to split
        self.trees = []
        self.initial_prediction = None

    def _get_loss_gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradient and hessian for the squared error loss function.
        l(y, y_hat) = 0.5 * (y - y_hat)^2
        """
        # Gradient (g) = d(loss)/d(y_pred) = y_pred - y_true
        gradient = y_pred - y_true
        # Hessian (h) = d^2(loss)/d(y_pred)^2 = 1
        hessian = np.ones_like(y_true)
        return gradient, hessian

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the XGBoost model.
        """
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(y.shape, self.initial_prediction)

        for i in range(self.n_estimators):
            # Calculate gradients (1st and 2nd order)
            gradients, hessians = self._get_loss_gradients(y, current_predictions)

            # Build a new tree to fit the gradients
            tree = Tree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                lambda_reg=self.lambda_reg,
                gamma_reg=self.gamma_reg
            )
            tree.fit(X, gradients, hessians)
            self.trees.append(tree)

            # Update predictions with the output of the new tree
            update = tree.predict(X)
            current_predictions += self.learning_rate * update

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions for a new set of data.
        """
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
