# Decision Tree Classifier

import numpy as np

### Node Class

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

### Splitting the data Function

def split_data(X, y, feature, threshold):
    # Create empty lists for left and right groups
    X_left, y_left = [], []
    X_right, y_right = [], []

    # Loop through each row in X
    for i in range(len(X)):
        # If value in the feature column is <= threshold add to left
        if X[i][feature] <= threshold:
            X_left.append(X[i])
            y_left.append(y[i])
        # Else add to right
        else:
            X_right.append(X[i])
            y_right.append(y[i])

    # Convert to numpy arrays
    import numpy as np
    X_left, y_left = np.array(X_left), np.array(y_left)
    X_right, y_right = np.array(X_right), np.array(y_right)

    return X_left, y_left, X_right, y_right

### Calculate the splitting criterion (Gini) function

def gini_impurity(y):
    
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

### Finding the best splits function

def find_best_split(X, y):
    # Track the best split
    best_feature = None
    best_threshold = None
    best_gini = float('inf')  # Start with a high value

    # Loop through feature indices
    for feature_idx in range(X.shape[1]):
        # Calculate the median value of feature
        feature_values = X[:, feature_idx]
        threshold = np.median(feature_values)  # Use median as threshold

        # Split data
        X_left, y_left, X_right, y_right = split_data(X, y, feature_idx, threshold)

        # Calculate Gini for left and right groups
        left_gini = gini_impurity(y_left)
        right_gini = gini_impurity(y_right)

        # Calculate weighted Gini Impurity
        total_samples = len(y_left) + len(y_right)
        weighted_gini = (
            (len(y_left) / total_samples) * left_gini
            + (len(y_right) / total_samples) * right_gini
        )

        # Check if this is the best split
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_feature = feature_idx
            best_threshold = threshold

    # Return the best split
    return best_feature, best_threshold, best_gini
    
### Build the tree function

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    # Stop if depth exceeds max_depth
    if max_depth is not None and depth >= max_depth:
        majority_class = max(set(y), key=list(y).count)
        return Node(value=majority_class)
    
    # Stop if the number of samples is too small
    if len(y) < min_samples_split:
        majority_class = max(set(y), key=list(y).count)
        return Node(value=majority_class)
    
    # Stopping condition
    if len(set(y)) == 1:
        return Node(value=y[0])
    
    # Find the best split
    best_feature, best_threshold, best_gini = find_best_split(X, y)
    
    # Create node
    node = Node(feature=best_feature, threshold=best_threshold)
    
    # Split data
    X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
    
    # Stop if no valid split
    if len(y_left) == 0 or len(y_right) == 0:
        majority_class = max(set(y), key=list(y).count)
        return Node(value=majority_class)
    
    # Build left and right child nodes
    node.left = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split)
    
    return node


### Predict

def predict(node, sample):
    # If this is a leaf node return value
    if node.value is not None:
        return node.value
        
    # Decide left or right based on the feature and threshold
    if sample[node.feature] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)