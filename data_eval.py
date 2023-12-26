"""
Machine learning algorithm evaluation functions. 

NAME: <Arvand Marandi>
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import randint
import math

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """

    size = table.row_count() # D rows
    training_set = DataTable(table.columns())
    test_set = table.copy()

    for i in range(size):
        rand_index = randint(0, size - 1)
        training_set.append(table[rand_index].values())

    for row in training_set:
        for i, r in enumerate(test_set):
            if row == r:
                del test_set[i]

    return (training_set, test_set)


def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    labels, counts = frequencies(table, label_col)
    proportions = []
    fraction = test_set_size / table.row_count()
    
    for index, label in enumerate(labels):
        calc = math.floor(counts[index] * fraction)
        proportions.append([label, calc])

    training_set = table.copy() 
    test_set = DataTable(table.columns())
    size = table.row_count() - 1

    for index, label in enumerate(labels): # for each label
        # Parse through the training set, randomly choosing rows with the current label until the proportion is met
        proportion = proportions[index][1] # proportion for the current label
        while proportion > 0:
            rand_index = randint(0, size - 1)
            if training_set[rand_index][label_col] == label:
                test_set.append(training_set[rand_index].values())
                del training_set[rand_index]
                size += -1
                proportion += -1

    return [training_set, test_set]

def tdidt_eval_with_tree(dt_root, test, label_col, labels):
    """Evaluates the given test set using tdidt over the dt_root, returning a corresponding confusion matrix.

    Args:
       dt_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Creating the matrix

    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    for row in test:
        actual = row[label_col]
        results = tdidt_predict(dt_root, row) # get the predictions for the current instance in the test set over the test set
        if results == None:
            predict_label = actual
        else:
            predict_label = results[0]
        
        for i in confusion_matrix: # increment the count of the prediction in the confusion matrix
            if i['actual'] == actual:
                i[predict_label] += 1

    return confusion_matrix

def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    remainder, test_set = stratified_holdout(remainder, label_col, (remainder.row_count() * (1/3)))
    trees = [] # list of pairs (tree, accuracy)

    labels = tuple(set(column_values(table, label_col))) # get the set of all labels in the training set

    for i in range(N):
        train, validation = 0, 0
        # Create boostrap samples
        boot = bootstrap(remainder)
        while boot[1].row_count() == 0:
            boot = bootstrap(remainder)
        train = boot[0]
        validation = boot[1]

        # Build a decision tree
        root = tdidt_F(train, label_col, F, columns)
        cleaned_tree = resolve_attribute_values(root, table)
        cleaned_tree = resolve_leaf_nodes(cleaned_tree)

        # Test the decision tree and produce resulting confusion matrix
        curr_matrix = tdidt_eval_with_tree(cleaned_tree, validation, label_col, labels)

        # Calculate the accuracy of the decision tree
        accuracies = []
        for label in labels:
            accuracies.append(accuracy(curr_matrix, label))

        curr_accuracy = sum(accuracies) / len(accuracies)
        # Add the tree and accuracy to the list of trees
        trees.append((cleaned_tree, curr_accuracy))

    trees.sort(key=lambda x: x[1], reverse=True) # sort the list of trees by accuracy

    top_trees = trees[:M] # return the top M trees

    m_trees = []

    # Calculate the accuracy of the top trees
    for tree in top_trees:
        curr_matrix = tdidt_eval_with_tree(tree[0], test_set, label_col, labels)
        accuracies = []
        for label in labels:
            accuracies.append(accuracy(curr_matrix, label))
        curr_accuracy = sum(accuracies) / len(accuracies)
        m_trees.append((tree[0], curr_accuracy))

    return m_trees

def random_forest_eval(table, train, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        train: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    # Creating the matrix

    labels = tuple(set(column_values(table, label_col))) # get the set of all labels in the training set
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    # Build the random forest
    forest = random_forest(table, train, F, M, N, label_col, columns) 
    
    # Filling the matrix
    for row in test:
        actual = row[label_col]
        predictions = []

        for tree in forest:
            result = tdidt_predict(tree[0], row)
            if result == None:
                pass
            else:
                predictions.append(result[0])

        if len(predictions) == 0:
            predictions.append(actual)

        predict_label = max(set(predictions), key=predictions.count) # get the prediction label with the highest count

        for i in confusion_matrix: # increment the count of the prediction in the confusion matrix
            if i['actual'] == actual:
                i[predict_label] += 1

    return confusion_matrix

def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """

    # Creating the matrix

    labels = tuple(set(column_values(train, label_col))) # get the set of all labels in the training set
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    # Filling the matrix

    root = tdidt(train, label_col, columns) # get the root of the decision tree
    cleaned_tree = resolve_attribute_values(root, train) 
    cleaned_tree = resolve_leaf_nodes(cleaned_tree) 

    for row in test:
        actual = row[label_col]
        results = tdidt_predict(cleaned_tree, row) # get the predictions for the current instance in the test set over the train set
        if results == None:
            predict_label = actual
        else:
            predict_label = results[0]
        
        for i in confusion_matrix: # increment the count of the prediction in the confusion matrix
            if i['actual'] == actual:
                i[predict_label] += 1

    return confusion_matrix

def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """

    labels = tuple(set(column_values(table, label_col)))
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    folds = stratify(table, label_col, k_folds)

    for i, curr in enumerate(folds):
        train_set = [fold for j, fold in enumerate(folds) if j != i]
        train_set = union_all(train_set)
        pred = tdidt_eval(train_set, curr, label_col, columns)
        values = []

        for row in pred: # parse through the vote matrix
            row_values = []
            for l in row.columns():
                if l != 'actual':
                    row_values.append(row[l])
            values.append(row_values)

        for i1, r in enumerate(confusion_matrix): # parse through the confusion matrix, where r is the row
            curr = values[i1]
            for i2, lab in enumerate(r.columns()): # where i2 is the index and lab is the label
                if lab != 'actual':
                    r[lab] += curr[i2 - 1]

    return confusion_matrix

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    stratified = []
    for i in range(k):
        stratified.append(DataTable(table.columns()))

    labels = distinct_values(table, label_column) # get the set of all labels in the table

    for label in labels: # for each label
        i = 1 # current fold
        for row in table: # for each row in the table
            if row[label_column] == label: # if the current row has the current label
                stratified[i-1].append(row.values()) # add the row to the current fold
                if i % k == 0: # if we are on the last fold, reset i to 1 (the first fold)
                    i = 1
                else: # otherwise, increment i
                    i += 1

    return stratified

def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """

    if len(tables) == 0:
        raise ValueError('The list of tables must be non-empty')
    
    union = DataTable(tables[0].columns()) # create a new table with the columns of the first table in the list of tables

    for table in tables: # for each table in the list of tables
        if table.columns() != union.columns(): # if the columns of the current table don't match the columns of the union table
            raise ValueError('The tables must have the exact same columns')
        for row in table:
            union.append(row.values()) # add the row to the union table

    return union

def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Creating the matrix

    labels = tuple(set(column_values(train, label_col))) # get the set of all labels in the training set
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    # Filling the matrix

    for row in test:
        predict_labels, probability = naive_bayes(train, row, label_col, continuous_cols, categorical_cols) # get the predictions for the current instance in the test set over the train set
        actual = row[label_col]

        for i in confusion_matrix: # increment the count of the prediction in the confusion matrix
            if i['actual'] == actual:
                i[predict_labels[0]] += 1

    return confusion_matrix


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    labels = tuple(set(column_values(table, label_col)))
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    folds = stratify(table, label_col, k_folds)

    for i, curr in enumerate(folds):
        train_set = [fold for j, fold in enumerate(folds) if j != i]
        train_set = union_all(train_set)
        pred = naive_bayes_eval(train_set, curr, label_col, cont_cols, cat_cols)

        values = []

        for row in pred: # parse through the vote matrix
            row_values = []
            for l in row.columns():
                if l != 'actual':
                    row_values.append(row[l])
            values.append(row_values)

        for i1, r in enumerate(confusion_matrix): # parse through the confusion matrix, where r is the row
            curr = values[i1]
            for i2, lab in enumerate(r.columns()): # where i2 is the index and lab is the label
                if lab != 'actual':
                    r[lab] += curr[i2 - 1]

    return confusion_matrix


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    labels = tuple(set(column_values(table, label_col)))
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    folds = stratify(table, label_col, k_folds)

    for i, curr in enumerate(folds):
        train_set = [fold for j, fold in enumerate(folds) if j != i]
        train_set = union_all(train_set)
        vote = knn_eval(train_set, curr, vote_fun, k, label_col, num_cols, nom_cols)

        values = []

        for row in vote: # parse through the vote matrix
            row_values = []
            for l in row.columns():
                if l != 'actual':
                    row_values.append(row[l])
            values.append(row_values)

        for i1, r in enumerate(confusion_matrix): # parse through the confusion matrix, where r is the row
            curr = values[i1]
            for i2, lab in enumerate(r.columns()): # where i2 is the index and lab is the label
                if lab != 'actual':
                    r[lab] += curr[i2 - 1]

    return confusion_matrix

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    size = table.row_count() - 1
    training_set = table.copy() 
    test_set = DataTable(table.columns())
    for i in range(test_set_size):
        rand_index = randint(0, size)
        test_set.append(training_set[rand_index].values()) # add the randomly selected row to the test set
        del training_set[rand_index] # delete the randomly selected row from the training set
        size += -1

    return [training_set, test_set]

def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    labels = tuple(set(column_values(train, label_col))) # get the set of all labels in the training set
    confusion_matrix = DataTable(['actual'] + [item for item in labels])
    for label in labels:
        confusion_matrix.append([label] + [0] * len(labels)) # add a row for each label to the confusion matrix

    for row in test:
        neighbours = knn(train, row, k, numeric_cols, nominal_cols) # get the k nearest neighbours for the current row
        values = [] 
        for instances in neighbours.values(): # convert the dictionary of neighbours to a list of the instances of the neighbours
            values.append(instances[0])
        
        scores = []
        distances = []
        for instances in neighbours.keys(): # get the distances of the neighbours
            distances.append(instances)
        max_dis = max(distances)
        min_dis = min(distances)
        for instances in distances:
            curr = (instances - min_dis) / (max_dis - min_dis) # normalize the distance with respect to the max and min distances
            scores.append(1 - curr)

        predictions = vote_fun(values, scores, label_col) # get the predictions for the current row
        
        actual = row[label_col]

        for i in confusion_matrix: # increment the count of the prediction in the confusion matrix
            if i['actual'] == actual:
                i[predictions[0]] += 1

    return confusion_matrix

def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    true_positive = 0
    for row in confusion_matrix:
        if row['actual'] == label:
            true_positive = row[label] 

    neg_labels = confusion_matrix.columns() # get the labels that aren't the label we're looking for
    neg_labels.remove('actual')
    neg_labels.remove(label)

    true_negative = 0
    for row in confusion_matrix:
        for l in neg_labels:
            if row['actual'] != label: # if the current label isn't the label we're looking for
                true_negative += row[l] # add to the true negative count with all the values in the row that aren't the label we're looking for

    total = 0
    for row in confusion_matrix:
        for l in confusion_matrix.columns():
            if l != 'actual':
                total += row[l]
    return (true_positive + true_negative) / total

def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    true_positive = 0
    for row in confusion_matrix:
        if row['actual'] == label:
            true_positive = row[label]
    
    false_positive = 0
    for row in confusion_matrix:
        if row['actual'] != label: # if the current label isn't the prediction label
            false_positive += row[label] # add to the false positive count with the number of false positive predictions

    return 0 if true_positive == 0 and false_positive == 0 else true_positive / (true_positive + false_positive)

def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    true_positive = 0
    for row in confusion_matrix:
        if row['actual'] == label:
            true_positive = row[label]

    false_negative = 0
    for row in confusion_matrix:
        if row['actual'] == label: # if the current label is the prediction label
            for l in confusion_matrix.columns():
                if l != label and l != 'actual': # if the current label isn't the prediction label and isn't the 'actual' label
                    false_negative += row[l] # add to the false negative count with the number of false negative predictions

    return true_positive / (true_positive + false_negative)