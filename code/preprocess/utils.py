import pickle
import os
from sklearn.metrics import mutual_info_score
import random


def load_existing_dataset(data_dir, category):
    reviews_pattern = os.path.join(data_dir, "X_" + category + "_5.pkl")
    labels_pattern = os.path.join(data_dir, "y_" + category + "_5.pkl")
    with open(reviews_pattern, 'rb') as f:
        reviews = pickle.load(f)
    with open(labels_pattern, 'rb') as f:
        labels = pickle.load(f)
    X_train, y_train, X_test, y_test = [], [], [], []
    pos_count = 0
    neg_count = 0
    for i in range(len(reviews)):
        if labels[i] == 0:
            if neg_count % 2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
            else:
                X_test.append(reviews[i])
                y_test.append(labels[i])
            neg_count += 1
        elif labels[i] == 1:
            if pos_count % 2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
            else:
                X_test.append(reviews[i])
                y_test.append(labels[i])
            pos_count += 1
    return X_train, y_train, X_test, y_test



def load_existing_dataset_and_create_splits(data_dir, category, train_size, val_size, test_size, shuffle=True):
    reviews_pattern = os.path.join(data_dir, "X_" + category + "_5.pkl")
    labels_pattern = os.path.join(data_dir, "y_" + category + "_5.pkl")
    with open(reviews_pattern, 'rb') as f:
        reviews = pickle.load(f)
    with open(labels_pattern, 'rb') as f:
        labels = pickle.load(f)
    assert len(reviews) >= (train_size+val_size+test_size), "you have less data than total number of splits you want to create"
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    train_pos_count = 0
    train_neg_count = 0
    val_pos_count = 0
    val_neg_count = 0
    test_pos_count = 0
    test_neg_count = 0
    for i in range(len(reviews)):
        if labels[i] == 0:
            if train_neg_count < train_size//2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
                train_neg_count += 1
            elif val_neg_count < val_size//2:
                X_val.append(reviews[i])
                y_val.append(labels[i])
                val_neg_count += 1
            elif test_neg_count < test_size//2:
                X_test.append(reviews[i])
                y_test.append(labels[i])
                test_neg_count += 1
            else:
                pass
        elif labels[i] == 1:
            if train_pos_count < train_size//2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
                train_pos_count += 1
            elif val_pos_count < val_size//2:
                X_val.append(reviews[i])
                y_val.append(labels[i])
                val_pos_count += 1
            elif test_pos_count < test_size//2:
                X_test.append(reviews[i])
                y_test.append(labels[i])
                test_pos_count += 1
            else:
                pass
    assert len(X_train) == train_size
    assert len(X_val) == val_size
    assert len(X_test) == test_size
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_file(path, file):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file


def get_sorted_loss(src_category, category_list, loss, to_reverse):
    loss_list = []
    for category in category_list:
        if category != src_category:
            src_trg_relative_loss = loss[(src_category, category)]
            loss_list.append((category, src_trg_relative_loss))
    loss_list.sort(key=lambda x: x[1], reverse=to_reverse)
    print("The relative loss from {} is :".format(src_category))
    for i in range(len(loss_list)):
        print(loss_list[i])
    return loss_list


def get_counts(X, i):
    return (sum(X[:,i]))


def get_top_NMI(n, X, target):
    MI = []
    length = X.shape[1]
    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs, MI


def overlap(list1, list2):
    return list(set(list1) & set(list2))
