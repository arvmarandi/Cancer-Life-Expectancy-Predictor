"""
Machine learning algorithm implementations.

NAME: <Arvand Marandi>
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math

def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F: The number of columns to return.
        columns: The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns  list is just returned.

    """
    if F >= len(columns):
        return columns
    
    f_columns = set()

    while len(f_columns) < F:
        f_columns.add(columns[randint(0, len(columns) - 1)])

    return list(f_columns)



def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """

    # Base Cases
    # 1. If table is empty, return None
    if table.row_count() == 0:
        return None
    # 2. If all instances have the same label, return a leaf node with that label
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    # 3. If no more attributes to partition on, return leaves from current partition
    columns_values = set()
    for row in table:
        for c in columns:
            columns_values.add(row[c])

    if columns == [] or len(columns_values) == 1: # if there are no more attributes to partition on, or if there is only one value in the column
        return build_leaves(table, label_col)
    
    # Recursive Case
    # 1. Calculate the entropy for each attribute
    entropy_dict = calc_e_new(table, label_col, random_subset(F, columns))
    # 2. Choose the attribute with the lowest entropy (if tie, choose first)
    lowest_entropy = min(entropy_dict.keys())
    partition_attribute = entropy_dict[lowest_entropy][0]
    # 3. Partition the table on this column
    partitioned_table_list = partition(table, [partition_attribute]) # list of subtables
    # 4. Create attribute node and fill in value nodes (recursive calls on partition)
    attribute_node = AttributeNode(partition_attribute, {})
    attribute_len = set()
    for row in table:
        attribute_len.add(row[partition_attribute])
    if len(attribute_len) == 1:
        return build_leaves(table, label_col)
    for subtable in partitioned_table_list:
        attribute_node.values[subtable[0][partition_attribute]] = tdidt_F(subtable, label_col, F, columns)
    # 5. Return the attribute node
    return attribute_node

def closest_centroid(centroids, row, columns):
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids: The list of rows serving as cluster centroids.
        row: The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    closest = dict() # key = distance, value = index of centroid

    for i, centroid in enumerate(centroids):
        distance = 0
        for c in columns:
            distance += (row[c] - centroid[c])**2
        if distance not in closest.keys():
            closest[distance] = []
        closest[distance].append(i)

    return closest[min(closest.keys())][0]

def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    if k > table.row_count():
        ValueError("k must be less than or equal to the number of rows in the table.")

    copy = table.copy()
    centroids = []
    size = copy.row_count()

    for i in range(k):
        rand_index = randint(0, size - 1)
        centroids.append(copy[rand_index])
        del copy[rand_index]
        size += -1

    return centroids

def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    # 1. Assign each instance to the cluster of its nearest centroid
    clusters = []

    for i in range(len(centroids)):
        clusters.append(DataTable(table.columns()))

    for row in table:
        index = closest_centroid(centroids, row, columns)
        clusters[index].append(row.values())

    # 2. Calculate the new centroids for each cluster
    new_centroids = []
    for cluster in clusters:
        values = []

        if cluster.row_count() == 0:
            new_centroids.append(centroids[clusters.index(cluster)])
            continue

        for c in table.columns():
            if c in columns:
                values.append((mean(cluster, c)))
            else:
                values.append(table[0][c])

        new_centroids.append(DataRow(table.columns(), values))

    # 3. If the new centroids are the same as the old centroids, return the clusters
    flag = True
    for i in range(len(centroids)):
        if centroids[i] != new_centroids[i]:
            flag = False
            break
    
    if flag:
        return clusters

    # 4. Else, repeat the process with the new centroids
    else:
        return k_means(table, new_centroids, columns)

def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list of data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    tss_list = []

    count = 0
    for cluster in clusters:
        count += 1
        total = 0
        for row in cluster:
            for c in columns:
                total += (row[c] - mean(cluster, c))**2
        tss_list.append(total)

    return tss_list

def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    label_set = set()

    for row in table:
        label_set.add(row[label_col])
    
    if len(label_set) == 1:
        return True
    
    return False


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    leaf_dict = {} # key = label, value = count

    for row in table:
        if row[label_col] not in leaf_dict.keys():
            leaf_dict[row[label_col]] = 0
        leaf_dict[row[label_col]] += 1

    leaves = [] # list of LeafNode objects

    for key, value in leaf_dict.items():
        leaves.append(LeafNode(key, value, table.row_count())) # create a LeafNode object for each label in the table

    return leaves

def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """
    # How to calculate entropy:
    # 1. Split the table into subtables based on the unique values in the column
    # 2. For each subtable, calculate the entropy for each unique value of the label column.
    # 3. Sum these entropy values and multiple by -1 to find the entropy for that column.
    # 4. Repeat for each column.

    entropy_dict = {} # key = entropy value, value = list of attributes with that entropy value

    for c in columns:
        values = distinct_values(table, c)

        weighted_average = 0 # entropy for the column
        entropy_values = {} # key = entropy value, value = list of counts for that entropy value

        # Split the table into subtables based on the unique values in the column
        for v in values: # 8
            subtable = DataTable(table.columns())
            value_entropy = 0 # entropy for the value

            for row in table:
                if row[c] == v:
                    subtable.append(row.values())

            # Calculate the entropy for each unique value of the label column
            labels = distinct_values(subtable, label_col)
            for l in labels:
                count = 0
                for row in subtable:
                    if row[label_col] == l:
                        count += 1
                p = count / subtable.row_count() # probability of the label in label column
                curr_entropy = p * math.log(p, 2) # entropy for the label
                value_entropy += curr_entropy

            if value_entropy != 0:
                value_entropy *= -1

            if value_entropy not in entropy_values.keys():
                entropy_values[value_entropy] = []
            entropy_values[value_entropy].append(subtable.row_count()) # for each unique value of the label column, add the count to the list of counts for that entropy value

        for key, value in entropy_values.items(): # calculate the weighted average
            for v in value:
                weighted_average += (v / table.row_count()) * key

        if weighted_average not in entropy_dict.keys():
            entropy_dict[weighted_average] = []
        entropy_dict[weighted_average].append(c)

    return entropy_dict

def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # Base Cases
    # 1. If table is empty, return None
    if table.row_count() == 0:
        return None
    # 2. If all instances have the same label, return a leaf node with that label
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    # 3. If no more attributes to partition on, return leaves from current partition
    columns_values = set()
    for row in table:
        for c in columns:
            columns_values.add(row[c])

    if columns == [] or len(columns_values) == 1: # if there are no more attributes to partition on, or if there is only one value in the column
        return build_leaves(table, label_col)
    
    # Recursive Case
    # 1. Calculate the entropy for each attribute
    entropy_dict = calc_e_new(table, label_col, columns)
    # 2. Choose the attribute with the lowest entropy (if tie, choose first)
    lowest_entropy = min(entropy_dict.keys())
    partition_attribute = entropy_dict[lowest_entropy][0]
    # 3. Partition the table on this column
    partitioned_table_list = partition(table, [partition_attribute]) # list of subtables
    # 4. Create attribute node and fill in value nodes (recursive calls on partition)
    attribute_node = AttributeNode(partition_attribute, {})
    attribute_len = set()
    for row in table:
        attribute_len.add(row[partition_attribute])
    if len(attribute_len) == 1:
        return build_leaves(table, label_col)
    for subtable in partitioned_table_list:
        attribute_node.values[subtable[0][partition_attribute]] = tdidt(subtable, label_col, columns)
    # 5. Return the attribute node
    return attribute_node

def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    summary_dict = {} # key = label, value = count

    # Base Case
    if type(dt_root) == LeafNode: # if dt_root is a LeafNode object
        if dt_root.label not in summary_dict.keys():
            summary_dict[dt_root.label] = 0
        summary_dict[dt_root.label] += dt_root.count
        return summary_dict
    
    # Recursive Case
    for key, value in dt_root.values.items():
        if type(value) == list: # if value is a list of LeafNode objects and not an AttributeNode object
            for leaf in value:
                if leaf.label not in summary_dict.keys():
                    summary_dict[leaf.label] = 0
                summary_dict[leaf.label] += leaf.count
        else: # if value is an AttributeNode object
            summary_dict = summarize_instances(value) # parse through the subtree

    return summary_dict

def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    # Base Cases
    # 1. If dt_root is a leaf node, return a copy of dt_root
    if type(dt_root) == LeafNode:
        return dt_root
    # 2. If dt_root is a list of leaf nodes, return a copy of the list and leaf nodes
    if type(dt_root) == list:
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]
    

    # Recursive Case
    # 1. Create a new decision tree attribute node
    new_dt_root = AttributeNode(dt_root.name, {})
    # 2. Recursively navigate through the tree
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_leaf_nodes(child)
    
    # Backtracking phase
    # 1. For each new_dt_root value, combine its leaves if it has multiple leaves
    for val, child in new_dt_root.values.items():
        if type(child) == list:
            max_count = 0
            max_leaf = None
            for leaf in child:
                if leaf.count > max_count:
                    max_count = leaf.count
                    max_leaf = leaf
            new_dt_root.values[val] = [max_leaf]

    return new_dt_root

def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    # Base Cases
    # 1. If dt_root is a leaf node, return a copy of dt_root
    if type(dt_root) == LeafNode:
        return dt_root
    # 2. If dt_root is a list of leaf nodes, return a copy of the list and leaf nodes
    if type(dt_root) == list:
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]

    # Recursive Case
    # 1. Create a new decision tree attribute node
    new_dt_root = AttributeNode(dt_root.name, {})
    # 2. Recursively navigate through the tree
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_attribute_values(child, table)

    # Backtracking phase
    values = distinct_values(table, dt_root.name)
    
    total = 0
    for val, child in new_dt_root.values.items():
        if type(child) == list:
            for leaf in child:
                total += leaf.count

    for count, v in enumerate(values): # for each value in the column
        if v not in new_dt_root.values.keys(): # if the attribute value is missing from the tree
            if count == len(values) - 1:
                summary = summarize_instances(new_dt_root)
                new_dt_root = []
                for key, value in summary.items():
                    new_dt_root.append(LeafNode(key, value, total))            
            else:
                # Replace new_dt_root with a new subtree that doesn't have the missing attribute value
                temp = AttributeNode(new_dt_root.name, {})
                for val, child in new_dt_root.values.items():
                    if val != v:
                        temp.values[val] = child
                new_dt_root = temp
    return new_dt_root

def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    # Base Cases
    if type(dt_root) == LeafNode:
        return dt_root.label, dt_root.percent()
    if type(dt_root) == list:
        return dt_root[0].label, dt_root[0].percent()
    
    # Recursive Case
    for val, child in dt_root.values.items():
        if instance[dt_root.name] == val:
            return tdidt_predict(child, instance)

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """
    labels = distinct_values(table, label_col)
    label_probabilities = {} # key = label, value = probability
    columns = continuous_cols + categorical_cols # list of all columns being considered

    #Calculate the probability of each label... P(C)
    p_c = {} # key = label, value = probability
    for row in table: 
        label = row[label_col]
        if label not in p_c.keys():
            p_c[label] = 0
        p_c[label] += 1

    for label in labels:
        # Calculate the probability of each column... P(X|C)
        p_x_given_c = 1
        
        label_table = DataTable(table.columns())
        for row in table:
            if row[label_col] == label:
                label_table.append(row.values())

        for c in columns:
            if c in continuous_cols:
                p_x_given_c *= gaussian_density(instance[c], mean(label_table, c), std_dev(label_table, c))
            else:
                # Find the number of instances in the column with the value of instance[c]
                count = 0
                for row in label_table:
                    if row[c] == instance[c]:
                        count += 1
                p_x_given_c *= count / p_c[label]

        # Bayes theorem: P(C|X) = P(X|C) * P(C)
        label_probabilities[label] = p_x_given_c * (p_c[label] / table.row_count())

    highest = max(label_probabilities.values())
    highest_labels = []
    for curr_label, prob in label_probabilities.items():
        if prob == highest:
            highest_labels.append(curr_label)

    return highest_labels, highest

def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    first, second = 0, 0

    if sdev > 0:
        first = 1 / ( math . sqrt (2 * math . pi ) * sdev ) # 1 / sqrt(2pi * sdev)
        second = math . e ** ( -(( x - mean ) ** 2) / (2 * ( sdev ** 2))) # e ^ -((x - mean)^2 / 2(sdev^2))
    return first * second

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """

    # Check that the numerical and nominal columns are disjoint.
    for c in numerical_columns:
        if c in nominal_columns:
            raise ValueError("Numerical and nominal columns must be disjoint.")

    # Check that the numerical and nominal columns are valid for the table.
    for c in numerical_columns:
        if c not in table.columns():
            raise ValueError("Numerical column not found in table: " + c)
    for c in nominal_columns:
        if c not in table.columns():
            raise ValueError("Nominal column not found in table: " + c)
    
    # Compute distances for just the numerical columns
    distances = {}

    if nominal_columns == []:
        for row in table:
            distance = 0
            for c in numerical_columns:
                distance += (instance[c] - row[c])**2 # Euclidean Distance
            if distance not in distances.keys(): # If key not in dictionary, value at key is a list; do this to account for repeats, as repeat distances are NOT discarded, but instead taken into account in KNN
                distances[distance] = []
            distances[distance].append(row)
    
    # Compute if there are nominal columns
    else:
        for row in table:
            distance = 0
            for c in numerical_columns:
                distance += (instance[c] - row[c])**2 # Euclidean Distance
            for c in nominal_columns:
                if instance[c] != row[c]:
                    distance += 1
            if distance not in distances.keys(): # If key not in dictionary, value at key is a list; do this to account for repeats, as repeat distances are NOT discarded, but instead taken into account in KNN
                distances[distance] = []
            distances[distance].append(row)

    # Find k-nearest neighbours
    count = 0
    k_closest = {}
    sorted_distances = {key: val for key, val in sorted(distances.items(), key = lambda ele: ele[0])} # Sort distances in ascending order
    for key, value in sorted_distances.items():
        if count < k:
            k_closest[key] = value
            count += 1
        else:
            break
    return k_closest

def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    values = []
    for row in instances:
        values.append(row[labeled_column])

    distinct_values = set(values)

    my_dict = {}
    for i in distinct_values: # label
        for j in values: # instance
            if j == i: # if the current instance is one of the labels
                if i not in my_dict.keys():
                    my_dict[i] = 0
                my_dict[i] += 1 # increment the counter

    count_values = list(my_dict.values()) # get the counts
    count_values.sort(reverse=True) # sort the count values in descending order
    highest = count_values[0]
    majority = []
    for key, value in my_dict.items():
        if value == highest:
            majority.append(key)

    return majority

def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    label_score_dict = {}
    for i, n in enumerate(instances):
        key = n[labeled_column] # key = the label
        if key not in label_score_dict.keys():
            label_score_dict[key] = 0
        label_score_dict[key] += scores[i] # value = the sum of scores for that label

    values = list(label_score_dict.values())
    values.sort(reverse=True) # sort in descending orde
    largest_sum = values[0]
    heaviest = []
    for k, v in label_score_dict.items():
        if v == largest_sum:
            heaviest.append(k)