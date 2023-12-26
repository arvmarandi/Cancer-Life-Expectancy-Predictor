"""
Data utility functions for the SEER dataset.

NAME: Arvand Marandi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import DataTable, DataRow

def standard_binning(table, column, cut_points, names):
    """Discretize column values into the standard bins. This
    function modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.
        cut_points: The list of cut points to discretize the dataset.
        names: The bin names to use.

    """
    for i in table:
        if i[column] == 'Survival months':
            continue # skip the header
        val = int(i[column]) # the current instance value for the column
        for j in range(len(cut_points)): # find the bin that the value falls into
            if val <= cut_points[j] and j == 0: # the first bin
                i[column] = names[j]
            elif val <= cut_points[j] and val > cut_points[j-1]: # if the value falls within the range
                i[column] = names[j]
            elif val > cut_points[j] and j == len(cut_points) - 1:
                i[column] = names[j+1]
                
def combine_features(table, columns, values, new_column_name):
    """Given the columns of a table, combine the columns into a new column.

    Args:
        table: The table containing the columns to combine.
        columns: The columns to combine.
        values: The values to combine (key = original name, value = new name)
        new_column_name: The name of the new column.

    """
    cols = table.columns() # get the columns of the table
    new_table = DataTable(cols[:-2] + [new_column_name]) # create a new table with the new column

    for row in table: # iterate through the rows of the original table
        new_table_vals = []
        vals = []
        for column in cols: # get the values from the columns
            if column not in columns:
                new_table_vals.append(row[column]) # append the values to the new table
            else: # if the column is in the columns to combine
                vals.append(row[column])

        # combine the values
        for v in vals:
            if v in values.keys():
                new_table_vals.append(values[v])

        new_table.append(new_table_vals) # append the new row to the new table

    return new_table