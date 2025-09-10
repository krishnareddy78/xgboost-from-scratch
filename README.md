# Advanced XGBoost from Scratch

This repository provides a from-scratch implementation of the XGBoost algorithm in Python, focusing on the core mathematical principles that distinguish it from traditional Gradient Boosting. It is intended for those who wish to understand the mechanics behind second-order optimization, regularization, and the specific tree-building process that makes XGBoost so powerful.

This is not a toy example; it is built with a focus on code quality, testability, and clear documentation, reflecting the standards of a senior data scientist or ML engineer.

## Key Features Implemented

* **Second-Order Optimization:** Utilizes both the Gradient (first derivative) and Hessian (second derivative) of the loss function, derived from its Taylor expansion. This provides a more accurate approximation of the loss function.
* **L2 Regularization (`lambda`):** Implements the regularization term in the objective function to control model complexity and prevent overfitting.
* **Similarity Score & Gain:** Tree splits are determined by calculating a "similarity score" for each potential leaf and maximizing the "gain" (the improvement in the objective function), rather than simply minimizing MSE.
* **Minimum Child Weight (`min_child_weight`):** Controls tree pruning by ensuring the sum of Hessians in a child node exceeds a certain threshold, preventing the creation of leaves that are too specific.
* **Structured as a Package:** The code is organized into a proper Python package with unit tests.
* **CI/CD Integration:** Includes a GitHub Actions workflow to automatically run tests.

## Mathematical Foundation

The objective function in XGBoost at step `t` is a combination of the loss and regularization:

$$
\text{Obj}^{(t)} = \sum_{i=1}^{n} [l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))] + \Omega(f_t)
$$

Using a second-order Taylor expansion on the loss function, we can approximate this objective as:

$$
\text{Obj}^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

Where:
- $g_i$ is the gradient: $\partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$
- $h_i$ is the Hessian: $\partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$
- $w_j$ is the score of leaf $j$.

By solving for the optimal leaf weights and plugging them back into the objective function, we get the objective score for a given tree structure:

$$
\text{Obj} = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

The **Gain** for a split is then calculated as:

$$
\text{Gain} = \text{Similarity}_{\text{Left}} + \text{Similarity}_{\text{Right}} - \text{Similarity}_{\text{Root}}
$$

where the **Similarity Score** for any node is:
$$ \text{Similarity Score} = \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} $$

This `Gain` is what our decision tree uses to find the best split.

## How to Use

1.  **Clone and Install:**
    ```bash
    git clone [https://github.com/your-username/xgboost-from-scratch.git](https://github.com/your-username/xgboost-from-scratch.git)
    cd xgboost-from-scratch
    pip install -e .
    ```

2.  **Run Tests:**
    ```bash
    pip install -r requirements-dev.txt
    pytest
    ```

3.  **Explore the Example:**
    Check out the `examples/usage_comparison.ipynb` notebook to see the model in action and compare its performance.
