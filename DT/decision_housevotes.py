import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter
import math
import numpy as np

class node_in_decision_tree:
    def __init__(self, feature, break_point):

        # Non-Leaf node attributes
        self.left = None
        self.right = None
        self.break_point = break_point
        self.feature = feature

        # Leaf Node attributes
        # This value will be False for all Non-leaf nodes and True for all leaf nodes
        self.is_leaf_node = False
        # This value will be None for Non-leaf nodes and one of the possible label for leaf nodes
        self.predicted_label_at_leaf_level = None
        
class construct_decision_tree:
    def __init__(self, max_depth_of_tree = 2):
        self.max_depth_of_tree = max_depth_of_tree
        
    def training_tree_with_data(self, training_features, training_labels):
        self.tree = self.generate_decision_tree(training_features, training_labels)
    
    def generate_decision_tree(self, training_features, training_labels, depth=0):
        no_of_unique_labels_in_training_labels = len(Counter(training_labels).keys())
        
        # Criterion to declare if a node is a leaf node
        if depth >= self.max_depth_of_tree or no_of_unique_labels_in_training_labels == 1:
            leaf = node_in_decision_tree(None, None)
            leaf.is_leaf_node = True
            leaf.predicted_label_at_leaf_level = self.high_frequent_label(training_labels)
            return leaf
        
        # Finding the best feature and the best value from the data set to further split the data set
        best_feature, best_break_point = self.best_split(training_features, training_labels)
        
        # split on best feature and threshold
        left_branch_data_set = training_features[:, best_feature] < best_break_point
        right_branch_data_set = ~left_branch_data_set
        
        left = self.generate_decision_tree(training_features[left_branch_data_set], training_labels[left_branch_data_set], depth+1)
        right = self.generate_decision_tree(training_features[right_branch_data_set], training_labels[right_branch_data_set], depth+1)
        
        node = node_in_decision_tree(best_feature, best_break_point)
        node.right = right
        node.left = left
        
        return node
    
    def best_split(self, training_features, training_labels):
        best_information_gain = -1
        split_idx = -1
        split_break_point = -1
        number_of_features = training_features.shape[1]
        # Loop to find the best attribute to select as node in tree for further split
        for feature_column_number in range(number_of_features):
            feature_data_list = training_features[:, feature_column_number]
            unique_feature_list = list(set((feature_data_list)))
            
            for value in unique_feature_list:
                information_gain = self.information_gain(training_labels, feature_data_list, value)
                
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    split_break_point = value
                    split_idx = feature_column_number
                    
        return split_idx, split_break_point
    
    def calculate_entropy_using_information_gain(self, training_labels):
        unique_ele_set = set(training_labels)
        freq_training_labels = {}
        for ele in unique_ele_set:
            freq_training_labels[ele]=0
        for ele in training_labels:
            freq_training_labels[ele]+=1
        entropy = 0
        for ele in freq_training_labels.keys():
            temp = freq_training_labels[ele]/len(training_labels)
            entropy += temp*math.log(temp,2)
        entropy = -entropy

        return entropy
    
    def information_gain(self, training_labels, feature_data_list, threshold):
        parent_entropy = self.calculate_entropy_using_information_gain(training_labels)

        left_tree_split_data_list = feature_data_list < threshold
        right_tree_split_data_list = ~ left_tree_split_data_list
        # print("left_idx",left_idx)
        # print("right_idx",right_idx)
        len_of_training_labels = len(training_labels)
        len_left_tree_split_data_list, len_right_tree_split_data_list = len(training_labels[left_tree_split_data_list]), len(training_labels[right_tree_split_data_list])
        
        if len_left_tree_split_data_list == 0 or len_right_tree_split_data_list == 0:
            return 0
        
        left_entropy = self.calculate_entropy_using_information_gain(training_labels[left_tree_split_data_list])
        right_entropy = self.calculate_entropy_using_information_gain(training_labels[right_tree_split_data_list])
        
        child_entropy = (len_left_tree_split_data_list / len_of_training_labels) * left_entropy
        child_entropy += (len_right_tree_split_data_list / len_of_training_labels) * right_entropy
        
        gain = parent_entropy - child_entropy
        
        return gain
    
    def high_frequent_label(self, training_labels):
        unique_ele_set = set(training_labels)
        freq_training_labels = {}
        for ele in unique_ele_set:
            freq_training_labels[ele]=0
        for ele in training_labels:
            freq_training_labels[ele]+=1
        key = None
        value = 0 
        for ele in freq_training_labels:
            if freq_training_labels[ele]>value:
                value = freq_training_labels[ele]
                key = ele
        return key
    
    def predict_correctness_of_label(self, inputs):
        tree_node = self.tree
        
        while not tree_node.is_leaf_node:
            if inputs[tree_node.feature] < tree_node.break_point:
                tree_node = tree_node.left
            else:
                tree_node = tree_node.right
                
        return tree_node.predicted_label_at_leaf_level
    
if __name__ == "__main__":
    
    # List to store 100 testing predictions accuracy
    testing_accuracies= []
    training_accuracies = []
    number_of_times = 100
    for time in range(number_of_times):

        testing_label_predictions = []
        training_label_predictions = []
        
        # Reading data set
        dafa_frame = pd.read_csv('house_votes_84.csv')

        # Shuffling the data for more accurate results
        dafa_frame = shuffle(dafa_frame)

        # Seperate all the feature columns from the data set
        features = dafa_frame.iloc[:, 0:16].values

        # Seperate all the class labels from the data set
        class_labels = dafa_frame['target'].values

        # print(features)
        # print(class_labels)

        training_features, testing_features, training_labels, testing_labels = train_test_split(features, class_labels, test_size=0.2, random_state=42)
        decision_tree_from_training_set = construct_decision_tree()
        decision_tree_from_training_set.training_tree_with_data(training_features, training_labels)

        for row_data in testing_features:
            testing_label_predictions.append(decision_tree_from_training_set.predict_correctness_of_label(row_data))

        for row_data in training_features:
            training_label_predictions.append(decision_tree_from_training_set.predict_correctness_of_label(row_data))

        training_accuracy = accuracy_score(training_labels, training_label_predictions)
        testing_accuracy = accuracy_score(testing_labels, testing_label_predictions)

        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)

    training_mean=np.mean(training_accuracies)
    testing_mean=np.mean(testing_accuracies)
    training_std=np.std(training_accuracies)
    testing_std=np.std(testing_accuracies)
    print(training_mean)
    print(testing_mean)
    print(training_std)
    print(testing_std)
    # print(training_accuracies)
    # print(testing_accuracies)

    # Plotting Histograms
    plt.title('Histogram of Decision Tree accuracies')
    plt.figure(1)
    plt.xlabel("Accuracies of Training set")
    plt.ylabel("Frequency of accuracy of Training set")
    plt.hist(training_accuracies)
    plt.figure(2)
    plt.title('Histogram of Decision Tree accuracies')
    plt.xlabel("Accuracies of Testing set")
    plt.ylabel("Frequency of accuracy of Testing set")
    plt.hist(testing_accuracies)
    plt.show()
