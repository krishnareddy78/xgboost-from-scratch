import numpy as np
import pytest
from advanced_xgboost import XGBoost
from advanced_xgboost.tree import Tree

@pytest.fixture
def synthetic_data():
    """Create a simple synthetic dataset for testing."""
    X = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
    y = np.array([1, 2, 3, 4])
    return X, y

def test_xgboost_initialization():
    """Test the XGBoost model initializes with correct parameters."""
    model = XGBoost(n_estimators=50, learning_rate=0.05, max_depth=5)
    assert model.n_estimators == 50
    assert model.learning_rate == 0.05
    assert model.max_depth == 5

def test_loss_gradients():
    """Test the calculation of gradients and hessians."""
    model = XGBoost()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.5, 2.5, 2.5])
    # g = y_pred - y_true
    # h = 1
    expected_g = np.array([0.5, 0.5, -0.5])
    expected_h = np.array([1, 1, 1])
    g, h = model._get_loss_gradients(y_true, y_pred)
    np.testing.assert_array_almost_equal(g, expected_g)
    np.testing.assert_array_almost_equal(h, expected_h)

def test_model_fit_predict(synthetic_data):
    """Test if the model can fit and predict without errors."""
    X, y = synthetic_data
    model = XGBoost(n_estimators=2, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (4,)
    assert not np.isnan(preds).any()

def test_tree_similarity_score():
    """Test the similarity score calculation in the Tree."""
    tree = Tree(lambda_reg=1.0)
    G = 10
    H = 5
    # score = G^2 / (H + lambda) = 100 / (5 + 1)
    expected_score = 100 / 6
    assert tree._calculate_similarity_score(G, H) == pytest.approx(expected_score)
