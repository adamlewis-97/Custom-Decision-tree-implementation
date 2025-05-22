import time
import pandas as pd
import numpy as np
import psutil
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from decision_tree import Node, build_tree, predict

# Get tree depth function
def get_tree_depth(node):
    if not node or (not node.left and not node.right):
        return 0
    return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))

# Count nodes function

def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

# Performance Data Collection
def collect_performance_data(X, y):
    results = []

    # Hyperparameter Ranges
    max_depth_values = [2, 4, 6, 8, 10, 12, 15, 20]
    min_samples_split_values = [2, 4, 8, 16, 32, 64]
    size_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for size_fraction in size_fractions:
                # Subset the data
                subset_X = X[:int(len(X) * size_fraction)]
                subset_y = y[:int(len(y) * size_fraction)]

                # Memory Tracking
                process = psutil.Process()

                # My Implementation
                start_time = time.time()
                root = build_tree(subset_X, subset_y, max_depth=max_depth, min_samples_split=min_samples_split)
                my_tree_training_time = time.time() - start_time
                my_memory_usage = process.memory_info().rss / (1024 * 1024)
                my_tree_depth = get_tree_depth(root)
                my_tree_num_nodes = count_nodes(root)
                
                # Predictions for My Implementation
                my_predictions = [predict(root, subset_X[i]) for i in range(len(subset_X))]

                # Accuracy
                my_tree_correct = sum(1 for i in range(len(subset_X)) if my_predictions[i] == subset_y[i])
                my_tree_accuracy = my_tree_correct / len(subset_X)

                # Precision and Recall

                # Multiclass Precision and Recall
                unique_classes = np.unique(subset_y)

                # Initialise metrics
                precision_per_class = []
                recall_per_class = []

                for cls in unique_classes:
                    true_positives = sum(1 for i in range(len(subset_X)) if my_predictions[i] == cls and 
                                         subset_y[i] == cls)
                    false_positives = sum(1 for i in range(len(subset_X)) if my_predictions[i] == cls and 
                                          subset_y[i] != cls)
                    false_negatives = sum(1 for i in range(len(subset_X)) if my_predictions[i] != cls and 
                                          subset_y[i] == cls)
    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + 
                                          false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + 
                                          false_negatives) > 0 else 0
    
                    precision_per_class.append(precision)
                    recall_per_class.append(recall)

                # Use macro averaging
                my_precision = np.mean(precision_per_class)
                my_recall = np.mean(recall_per_class)
               
                # Sklearn implementation
                start_time = time.time()
                sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, 
                                                      min_samples_split=min_samples_split, random_state=0)
                sklearn_tree.fit(subset_X, subset_y)
                sklearn_training_time = time.time() - start_time
                sklearn_memory_usage = process.memory_info().rss / (1024 * 1024) 
                sklearn_depth = sklearn_tree.get_depth()
                sklearn_num_nodes = sklearn_tree.tree_.node_count
                sklearn_predictions = sklearn_tree.predict(subset_X)
                sklearn_accuracy = accuracy_score(subset_y, sklearn_predictions)
                sklearn_precision = precision_score(subset_y, sklearn_predictions, average='weighted', 
                                                    zero_division=0)
                sklearn_recall = recall_score(subset_y, sklearn_predictions, average='weighted', zero_division=0)

                # Store Results
                results.append({
                    "dataset_size": len(subset_X),
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "my_tree_accuracy": my_tree_accuracy,
                    "my_tree_precision": my_precision, 
                    "my_tree_recall": my_recall,        
                    "my_tree_depth": my_tree_depth,
                    "my_tree_num_nodes": my_tree_num_nodes,
                    "my_tree_training_time": my_tree_training_time,
                    "my_memory_usage": my_memory_usage,
                    "sklearn_accuracy": sklearn_accuracy,
                    "sklearn_precision": sklearn_precision,
                    "sklearn_recall": sklearn_recall,
                    "sklearn_tree_depth": sklearn_depth,
                    "sklearn_num_nodes": sklearn_num_nodes,
                    "sklearn_training_time": sklearn_training_time,
                    "sklearn_memory_usage": sklearn_memory_usage,
                })

    # Convert Results to DataFrame
    return pd.DataFrame(results)