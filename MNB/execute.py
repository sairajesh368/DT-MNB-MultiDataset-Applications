from utils import *
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

def construct_column_sum_dict(training_set):
    matrix_column_sum_dict = {}
    for document in training_set:
        for word in document:
            if word in matrix_column_sum_dict:
                matrix_column_sum_dict[word]+=1
            else:
                matrix_column_sum_dict[word]=1
    for word in vocab:
        if word not in matrix_column_sum_dict:
            matrix_column_sum_dict[word] = 0
    return matrix_column_sum_dict

def calculating_correct_predictions_count_without_log(testing_data_set, alpha, flag):

    correct_predictions_count = 0
    for sentence in testing_data_set:
        pr_positive_sentence = probability_positive_train
        pr_negative_sentence = probability_negative_train
        for word in sentence:
            if word in probability_positive_dict:
                pr_positive_sentence*=probability_positive_dict[word]
                pr_negative_sentence*=probability_negative_dict[word]
            else:
                pr_positive_sentence*=(alpha/(vocab_length*alpha))
                pr_negative_sentence*=(alpha/(vocab_length*alpha))
        if flag==1 and pr_positive_sentence > pr_negative_sentence:
            correct_predictions_count+=1
        if flag==0 and pr_negative_sentence > pr_positive_sentence:
            correct_predictions_count+=1
    return correct_predictions_count

def calculating_correct_predictions_count_with_log(testing_data_set, alpha, flag):

    correct_predictions_count = 0
    for sentence in testing_data_set:
        pr_positive_sentence = math.log(probability_positive_train)
        pr_negative_sentence = math.log(probability_negative_train)
        for word in sentence:
            if word in probability_positive_dict:
                pr_positive_sentence+=math.log(probability_positive_dict[word])
                pr_negative_sentence+=math.log(probability_negative_dict[word])
            else:
                pr_positive_sentence+=math.log(alpha/(vocab_length*alpha))
                pr_negative_sentence+=math.log(alpha/(vocab_length*alpha))
        if flag==1 and pr_positive_sentence > pr_negative_sentence:
            correct_predictions_count+=1
        if flag==0 and pr_negative_sentence > pr_positive_sentence:
            correct_predictions_count+=1
    return correct_predictions_count


def calculating_confusion_matrix_accuracy_recall_without_log(alpha):
    TP_without_log = calculating_correct_predictions_count_without_log(pos_test,alpha,1)
    TN_without_log = calculating_correct_predictions_count_without_log(neg_test,alpha,0)
    FN_without_log = pos_test_length - TP_without_log
    FP_without_log = neg_test_length - TN_without_log

    correct_predictions_count_without_log = TP_without_log + TN_without_log
    print("correct_predictions_count_without_log",correct_predictions_count_without_log)

    confusion_matrix_without_log = [[TP_without_log,FN_without_log],[FP_without_log,TN_without_log]]
    accuracy_without_log = correct_predictions_count_without_log/(pos_test_length+neg_test_length)
    precision_without_log = confusion_matrix_without_log[0][0]/(confusion_matrix_without_log[0][0]+confusion_matrix_without_log[1][0])
    recall_without_log = confusion_matrix_without_log[0][0]/(confusion_matrix_without_log[0][0]+confusion_matrix_without_log[0][1])

    print("Values without using Log")
    print("accuracy_without_log", accuracy_without_log)
    print("precision_without_log", precision_without_log)
    print("recall_without_log", recall_without_log)
    print("Confusion Matrix without log")
    print("TP:", confusion_matrix_without_log[0][0], "FN:", confusion_matrix_without_log[0][1])
    print("FP:", confusion_matrix_without_log[1][0], "TN:", confusion_matrix_without_log[1][1])
    print()

def calculating_confusion_matrix_accuracy_recall_with_log(alpha):

    TP_with_log = calculating_correct_predictions_count_with_log(pos_test,alpha,1)
    TN_with_log = calculating_correct_predictions_count_with_log(neg_test,alpha,0)
    FN_with_log = pos_test_length - TP_with_log
    FP_with_log = neg_test_length - TN_with_log

    correct_predictions_count_with_log = TP_with_log + TN_with_log
    print("correct_predictions_count_with_log",correct_predictions_count_with_log)

    confusion_matrix_with_log = [[TP_with_log,FN_with_log],[FP_with_log,TN_with_log]]
    accuracy_with_log = correct_predictions_count_with_log/(pos_test_length+neg_test_length)
    precision_with_log = 1
    recall_with_log = 0
    if confusion_matrix_with_log[0][0]!=0 and confusion_matrix_with_log[1][0]!=0:
        precision_with_log = confusion_matrix_with_log[0][0]/(confusion_matrix_with_log[0][0]+confusion_matrix_with_log[1][0])
    if confusion_matrix_with_log[0][0]!=0 and confusion_matrix_with_log[0][1]!=0:
        recall_with_log = confusion_matrix_with_log[0][0]/(confusion_matrix_with_log[0][0]+confusion_matrix_with_log[0][1])
    accuracies_list.append(accuracy_with_log)
    print("Values with using Log")
    print("accuracy_with_log", accuracy_with_log)
    print("precision_with_log", precision_with_log)
    print("recall_with_log", recall_with_log)
    print("Confusion Matrix with log")
    print("TP:", confusion_matrix_with_log[0][0], "FN:", confusion_matrix_with_log[0][1])
    print("FP:", confusion_matrix_with_log[1][0], "TN:", confusion_matrix_with_log[1][1])
    print()

def calculating_probability_of_each_word(positive_matrix_sum, negative_matrix_sum):
    pr_positive_dict = {}
    pr_negative_dict = {}
    for word in vocab:
        pr_positive_dict[word]=(positive_matrix_sum[word]/positive_probability_denominator)
        pr_negative_dict[word]=(negative_matrix_sum[word]/negative_probability_denominator)
    return pr_positive_dict, pr_negative_dict

def calculating_denominator(matrix_sum_dict, alpha):
    return (sum(matrix_sum_dict.values()) + (vocab_length * alpha))

def adding_alpha_to_column_sum_in_matrix(matrix_sum_alpha, alpha):
    for key in matrix_sum_alpha:
        matrix_sum_alpha[key]+=alpha
    return matrix_sum_alpha


question_number = (sys.argv[1].lower())
if(question_number == "q1" or question_number == "q2"):
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2
    percentage_positive_instances_test  = 0.2
    percentage_negative_instances_test  = 0.2
elif(question_number == "q3"):
    percentage_positive_instances_train = 1
    percentage_negative_instances_train = 1
    percentage_positive_instances_test  = 1
    percentage_negative_instances_test  = 1
elif(question_number == "q4"):
    percentage_positive_instances_train = 0.5
    percentage_negative_instances_train = 0.5
    percentage_positive_instances_test  = 1
    percentage_negative_instances_test  = 1
elif(question_number == "q6"):
    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5
    percentage_positive_instances_test  = 1
    percentage_negative_instances_test  = 1

(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

# print(pos_train)
# print(neg_train)

pos_train_length = len(pos_train)
neg_train_length = len(neg_train)
pos_test_length = len(pos_test)
neg_test_length = len(neg_test)
vocab_length = len(vocab)

print("Number of positive training instances:", pos_train_length)
print("Number of negative training instances:", neg_train_length)
print("Number of positive test instances:", pos_test_length)
print("Number of negative test instances:", neg_test_length)
print("Number of words in vocab:",vocab_length)

probability_positive_train=pos_train_length/(pos_train_length+neg_train_length)
probability_negative_train=neg_train_length/(pos_train_length+neg_train_length)
print("probability_positive_train",probability_positive_train)
print("probability_negative_train",probability_negative_train)


original_positive_matrix_sum_dict = construct_column_sum_dict(pos_train)
original_negative_matrix_sum_dict = construct_column_sum_dict(neg_train)


positive_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_positive_matrix_sum_dict, 1e-5)
negative_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_negative_matrix_sum_dict, 1e-5)


positive_probability_denominator = calculating_denominator(original_positive_matrix_sum_dict, 1e-5)
negative_probability_denominator = calculating_denominator(original_negative_matrix_sum_dict, 1e-5)


probability_positive_dict = {}
probability_negative_dict = {}
probability_positive_dict, probability_negative_dict = calculating_probability_of_each_word(positive_matrix_sum_dict, negative_matrix_sum_dict)

accuracies_list=[]
# # Q1
print()
calculating_confusion_matrix_accuracy_recall_without_log(1e-5)
calculating_confusion_matrix_accuracy_recall_with_log(1e-5)

accuracies_list = []
# # Q2
# alpha_values_list = []
# alpha_value = 0.0001
# for i in range(8):
#     print("alpha value",alpha_value)
#     alpha_values_list.append(alpha_value)
#     probability_positive_dict = {}
#     probability_negative_dict = {}
#     positive_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_positive_matrix_sum_dict, alpha_value)
#     negative_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_negative_matrix_sum_dict, alpha_value)
#     positive_probability_denominator = calculating_denominator(original_positive_matrix_sum_dict, alpha_value)
#     negative_probability_denominator = calculating_denominator(original_negative_matrix_sum_dict, alpha_value)
#     # print("positive_probability_denominator ",positive_probability_denominator)
#     # print("negative_probability_denominator ",negative_probability_denominator)
#     probability_positive_dict, probability_negative_dict = calculating_probability_of_each_word(positive_matrix_sum_dict, negative_matrix_sum_dict)
#     calculating_confusion_matrix_accuracy_recall_with_log(alpha_value)
#     alpha_value*=10


# print(accuracies_list)
# print(alpha_values_list)

# x = [1,2,3,4,5,6,7,8]
# plt.plot(x, accuracies_list)
# plt.xticks(x, alpha_values_list)
# # giving a title to my graph
# plt.title('Alpha-to-Accuracy-Graph')
# # naming the x axis
# plt.xlabel('Alpha values')
# # naming the y axis
# plt.ylabel('Accuracy with log probabilities')
# plt.grid()
# plt.show()



high_accuracy_alpha_value = 10
probability_positive_dict = {}
probability_negative_dict = {}
positive_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_positive_matrix_sum_dict, high_accuracy_alpha_value)
negative_matrix_sum_dict = adding_alpha_to_column_sum_in_matrix(original_negative_matrix_sum_dict, high_accuracy_alpha_value)
positive_probability_denominator = calculating_denominator(original_positive_matrix_sum_dict, high_accuracy_alpha_value)
negative_probability_denominator = calculating_denominator(original_negative_matrix_sum_dict, high_accuracy_alpha_value)
probability_positive_dict, probability_negative_dict = calculating_probability_of_each_word(positive_matrix_sum_dict, negative_matrix_sum_dict)

# # Q3
# print()
# print("alpha value for Q3 is 10")
# calculating_confusion_matrix_accuracy_recall_with_log(high_accuracy_alpha_value)

# # Q4
# print()
# print("alpha value for Q4 is 10")
# calculating_confusion_matrix_accuracy_recall_with_log(high_accuracy_alpha_value)

# # Q6
# print()
# print("alpha value for Q6 is 10")
# calculating_confusion_matrix_accuracy_recall_with_log(high_accuracy_alpha_value)






