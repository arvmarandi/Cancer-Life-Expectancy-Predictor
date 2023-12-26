"""
Data utility functions.

NAME: Arvand Marandi
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt

def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: The column in the table to normalize.

    """
    values = []
    for row in table:
        values.append(row[column])

    max_val = max(values)
    min_val = min(values)

    for row in table:
        row[column] = (row[column] - min_val) / (max_val - min_val) # normalize the value using x - min / max - min

def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.
        cut_points: The list of cut points to discretize the dataset.

    """
    ordinal = 1
    for i in range(table.row_count()):
        if i in cut_points:
            ordinal += 1 # increment the ordinal value if we have a cut point
        table[i][column] = ordinal

def column_values(table, column):
    """Returns a list of the values (in order) in the given column.

    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    values = []

    for i in table: # go through the rows in the table
        values.append(i[column]) # append the value to the list

    return values


def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from

    Notes: 
        Assumes there are no missing values in the column.

    """
    values = column_values(table, column) # get the values from the column
    return sum(values) / len(values) # return the mean

def variance(table, column):
    """Returns the variance of the values in the given table column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from

    Notes:
        Assumes there are no missing values in the column.

    """
    values = column_values(table, column)
    my_mean = mean(table, column)
    variance_sum = 0
    for i in values:
        variance_sum += (i - my_mean) ** 2 # sum up the squares of the difference between the value and the mean
    
    return variance_sum / len(values) # return the variance


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from

    Notes:
        Assumes there are no missing values in the column.

    """
    return sqrt(variance(table, column))


def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    xs = column_values(table, x_column)
    ys = column_values(table, y_column)
    x_mean = mean(table, x_column)
    y_mean = mean(table, y_column)
    covariance_sum = 0
    for i, n in enumerate(xs): # enumerate allows us to iterate through the list and keep track of the index
        covariance_sum += (n - x_mean) * (ys[i] - y_mean)  # sum up the product of the difference between the value and the mean for both columns
    return covariance_sum / len(xs)


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.

    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    """
    x_mean = mean(table, x_column)
    y_mean = mean(table, y_column)
    slope = (covariance(table, x_column, y_column))/(variance(table, x_column)) # slope is the covariance of x and y divided by the variance of x
    intercept = y_mean - (slope)*(x_mean) # y-intercept is the mean of y minus the slope times the mean of x
    return [slope, intercept]


def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.

    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    return (covariance(table, x_column, y_column))/(std_dev(table, x_column) * std_dev(table, y_column)) # correlation coefficient is the covariance of x and y divided by the product of the standard deviations of x and y


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range

    Notes:
        start must be less than end

    """
    if start >= end:
        raise ValueError("start must be less than end")

    values = column_values(table, column)
    c = 0
    for v in values:
        if v >= start and v < end: # if the value falls within the range
            c += 1 # increment the counter
    return c # return the counter


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure() # reset figure
    plt.hist(column_values(table, column), bins=nbins, color= "blue", rwidth=0.8) # create the histogram
    plt.xlabel(xlabel) # assign the x label
    plt.ylabel(ylabel) # assign the y label
    plt.title(title) # assign the title

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()
    

def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure() # reset figure
    plt.scatter(column_values(table, xcolumn), column_values(table, ycolumn)) # create the scatter plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    slope, intercept = linear_regression(table, xcolumn, ycolumn) # get the slope and intercept from the linear regression function
    best_fit_line = [slope * x + intercept for x in column_values(table, xcolumn)] # create the best fit line
    plt.plot(column_values(table, xcolumn), best_fit_line, color='red', label='Best Fit Line')

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    unique = set()
    for i in table:
        item = i[column]
        unique.add(item)

    return list(unique)

def remove_missing(table, columns, specifics=None):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.
        specifics: A dictionary of column names and specific missing values to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    new_table = DataTable(table.columns())

    # Verifying column names are valid
    for i in columns:
        if i not in table.columns():
            raise IndexError('bad column name')
        
    for i in table:
        no_missing = True
        for j in columns:
            if specifics is not None:
                if j in specifics.keys():
                    for k in specifics[j]:
                        if i[j] == k:
                            no_missing = False
                    if i[j] == "" or i[j] == "Unknown": 
                        no_missing = False # Set the flag to false
            if i[j] == "" or i[j] == "Unknown": # If we have a missing value
                no_missing = False # Set the flag to false
        if no_missing:
            new_table.append(i.values())

    return new_table

def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """

    new_table = DataTable(table.columns())
    instances = []
    dups = set()
    new_set = set() # This set will contain the duplicate instances, but ensures there are no duplicates of duplicates (if that makes sense)

    for i in table:
        instances.append(i.values())

    for instance in instances: # parsing through the list of instances 
        if tuple(instance) in dups: # tuples are utilized here because lists are not hashable, but tuples are. 
            new_set.add(tuple(instance)) 
        else:
            dups.add(tuple(instance))

    for i in new_set:
        new_table.append(list(i))

    return new_table
                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    
    new_table = DataTable(table.columns())
    instances = []
    cleaned_instances = set()

    for i in table:
        instances.append(i.values())

    for instance in instances:
        if tuple(instance) not in cleaned_instances: # tuples are utilized here because lists are not hashable, but tuples are.
            cleaned_instances.add(tuple(instance))

    for i in cleaned_instances:
        new_table.append(list(i))

    return new_table


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    
    clean_table = table.copy()
    clean_table.drop(list(set(table.columns()).difference(set(columns)))) # drop the columns that aren't being partitioned on
    clean_table = remove_duplicates(clean_table) # remove all duplicates
    clean_list = []
    for i in clean_table:
        clean_list.append(i.values())

    partition_list = [] # appending data tables to this list
    for i in clean_list:
        current = DataTable(table.columns())
        for j in table:
            if j.values(columns) == i:
                current.append(j.values())
        partition_list.append(current)

    return partition_list


def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """

    values = []
    for i in table:
        if i[column] != "":
            values.append(i[column]) # append the value to the list

    return function(values)


def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """

    replaced = table.copy() 

    for i in replaced:
        if i[column] == "": # if we have a missing value
            my_list = []
            for j in table:
                if (i.values(partition_columns) == j.values(partition_columns)) and (j[column] != ""): # if the partition columns match and the value is not missing
                    my_list.append(j[column]) # append the value to the list
            val = function(my_list)
            i[column] = val

    return replaced


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """

    summary_vals = []
    vals = []

    # create a set of values that fall under partition column
    partition_set = set() # this set will contain the unique values in the partition column
    for i in table:
        partition_set.add(i[partition_column])

    for i in partition_set: # for each unique value in the partition column
        for j in table: 
            if j[partition_column] == i: # if the partition column value matches the unique value
                vals.append(j[stat_column])
        summary_vals.append(function(vals))
    
    return list(partition_set), summary_vals


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """

    partition_set = set()
    frequency_map = {}
    frequency_list = []

    for i in table:
        partition_set.add(i[partition_column])

    for i in table:
        if i[partition_column] in partition_set:
            frequency_map[i[partition_column]] = frequency_map.get(i[partition_column], 0) + 1 # the values in the map are the frequency of the partition column value

    labels = []
    for i in frequency_map.keys():
        labels.append(i)

    for i in frequency_map:
        frequency_list.append(frequency_map[i])

    return labels, frequency_list
        


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """

    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """

    plt.figure() # reset figure
    plt.pie(values, labels=labels) # create the pie chart
    plt.title(title) # assign the title

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()


def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """

    plt.figure() # reset figure
    plt.bar(bar_names, bar_values) # create the bar chart
    plt.xlabel(xlabel) # assign the x label
    plt.ylabel(ylabel) # assign the y label
    plt.title(title) # assign the title

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()


def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """

    plt.figure() # reset figure
    plt.scatter(xvalues, yvalues) # create the scatter plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """

    plt.figure() # reset figure
    plt.boxplot(distributions) # create the box plot
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45) # assign the labels to the x axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()