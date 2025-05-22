import numpy as np

from decision_tree import split_data

def test_split_data():
    # Example input
    X = np.array([[2, 3], [6, 7]])
    y = np.array([0, 1])
    feature = 0
    threshold = 5.0

    # Expected output
    X_left_expected = np.array([[2, 3]])
    y_left_expected = np.array([0])
    X_right_expected = np.array([[6, 7]])
    y_right_expected = np.array([1])

    # Run the function
    X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)

    # Assertions
    assert np.array_equal(X_left, X_left_expected), "X_left is incorrect"
    assert np.array_equal(y_left, y_left_expected), "y_left is incorrect"
    assert np.array_equal(X_right, X_right_expected), "X_right is incorrect"
    assert np.array_equal(y_right, y_right_expected), "y_right is incorrect"


from decision_tree import gini_impurity

def test_gini_impurity():
    # Pure dataset
    y1 = [1, 1, 1, 1]
    assert gini_impurity(y1) == 0.0, "Gini impurity for pure dataset should be 0.0"

    # Equal distribution
    y2 = [1, 0, 1, 0]
    assert gini_impurity(y2) == 0.5, "Gini impurity for equal distribution should be 0.5"

    # Unbalanced distribution
    y3 = [1, 1, 1, 0]
    expected_gini = 1 - (3/4)**2 - (1/4)**2  # 0.375
    assert gini_impurity(y3) == expected_gini, "Gini impurity for unbalanced dataset is incorrect"


from decision_tree import find_best_split

def test_find_best_split():
    # Simple dataset
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    y = np.array([0, 0, 1, 1])

    # Run the function
    best_feature, best_threshold, best_gini = find_best_split(X, y)

    # Expected outputs
    expected_feature = 0  # The best split is on the first feature
    expected_threshold = 2.5  # Median of feature 0
    expected_gini = 0.0  

    # Assertions
    assert best_feature == expected_feature, "Incorrect feature chosen for the split"
    assert best_threshold == expected_threshold, "Incorrect threshold for the split"
    assert best_gini == expected_gini, "Incorrect Gini value for the best split"

from decision_tree import build_tree, predict

def test_build_tree():
    # Input dataset
    X = np.array([[2, 3], [6, 7], [8, 9], [10, 11]])
    y = np.array([0, 0, 1, 1])

    # Build tree with max depth 2
    root = build_tree(X, y, max_depth=2)

    # Test root node
    assert root.feature == 0, "Root node split should be on feature 0"
    assert root.threshold == 7.0, "Root node split should be at threshold 7.0"

    # Test left child
    left_child = root.left
    assert left_child.value == 0, "Left child should predict class 0"

    # Test right child
    right_child = root.right
    assert right_child.value == 1, "Right child should predict class 1"

def test_predict():
    # Build a simple tree
    X = np.array([[2, 3], [6, 7], [8, 9], [10, 11]])
    y = np.array([0, 0, 1, 1])
    root = build_tree(X, y, max_depth=2)

    # Test predictions
    assert predict(root, [2, 3]) == 0, "Prediction for [2, 3] should be 0"
    assert predict(root, [10, 11]) == 1, "Prediction for [10, 11] should be 1"

