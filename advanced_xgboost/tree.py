import numpy as np
from typing import Optional, Tuple

class Node:
    """A single node in a decision tree."""
    def __init__(self, value: Optional[float] = None, feature_index: Optional[int] = None,
                 threshold: Optional[float] = None, left=None, right=None, gain: Optional[float] = None):
        self.value = value  # The optimal weight if this is a leaf node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain

class Tree:
    """A decision tree for the XGBoost model."""
    def __init__(self, max_depth: int = 3, min_child_weight: int = 1,
                 lambda_reg: float = 1.0, gamma_reg: float = 0.0):
        self.root = None
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_reg = lambda_reg
        self.gamma_reg = gamma_reg

    def _calculate_similarity_score(self, G: float, H: float) -> float:
        """Calculate the similarity score for a set of gradients and hessians."""
        return np.power(G, 2) / (H + self.lambda_reg)

    def _calculate_leaf_value(self, G: float, H: float) -> float:
        """Calculate the optimal weight for a leaf node."""
        return -G / (H + self.lambda_reg)

    def _find_best_split(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Iterate through features and thresholds to find the best split."""
        best_gain = -np.inf
        best_feature_index, best_threshold = None, None
        n_samples, n_features = X.shape

        if n_samples < 2:
            return best_feature_index, best_threshold, best_gain
        
        G_root = np.sum(gradients)
        H_root = np.sum(hessians)
        root_similarity = self._calculate_similarity_score(G_root, H_root)

        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                left_mask = X[:, feature_index] < threshold
                
                G_left, H_left = np.sum(gradients[left_mask]), np.sum(hessians[left_mask])
                G_right, H_right = G_root - G_left, H_root - H_left

                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue
                
                left_similarity = self._calculate_similarity_score(G_left, H_left)
                right_similarity = self._calculate_similarity_score(G_right, H_right)
                
                gain = left_similarity + right_similarity - root_similarity
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, current_depth: int) -> Node:
        """Recursively build the tree."""
        G = np.sum(gradients)
        H = np.sum(hessians)

        # Conditions for creating a leaf node
        if current_depth >= self.max_depth or H < self.min_child_weight:
            return Node(value=self._calculate_leaf_value(G, H))

        feature_index, threshold, gain = self._find_best_split(X, gradients, hessians)

        if gain <= self.gamma_reg:
            return Node(value=self._calculate_leaf_value(G, H))

        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], gradients[left_mask], hessians[left_mask], current_depth + 1)
        right_child = self._build_tree(X[right_mask], gradients[right_mask], hessians[right_mask], current_depth + 1)
        
        return Node(feature_index=feature_index, threshold=threshold, gain=gain, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """Public method to start building the tree."""
        self.root = self._build_tree(X, gradients, hessians, 0)
    
    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        """Traverse the tree for a single data point."""
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for a dataset."""
        return np.array([self._predict_single(x, self.root) for x in X])
