""" 
Data Table implementation.

NAME: Arvand Marandi
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """
        if column not in self.columns(): # if column not in the list of columns
            raise IndexError('bad column name')
        index = self.columns().index(column) # get index of column
        del self.__columns[index] # delete column name
        del self.__values[index] # delete column value
        

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        for i in range(len(self.__columns)): # for each column
            if self.__columns[i] != other.__columns[i]: # if column names are not equal
                return False
        for i in range(len(self.__values)): # for each value
            if self.__values[i] != other.__values[i]: # if values are not equal
                return False
        return True
    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns] # return the values of the given columns as a list


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
        if columns is None: # if no columns given
            return DataRow(self.columns(), self.values()) # return the same DataRow
        return DataRow(columns, self.values(columns)) # return a new DataRow with the given columns and values
    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        return tabulate.tabulate([self.columns()] + [row.values() for row in self.__row_data], headers='firstrow') # return the table with the column names as the first row

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        if row_index >= len(self.__row_data): # if row_index is out of bounds
            raise IndexError('bad row index')
        del self.__row_data[row_index] # delete the row at row_index
        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        
        if(len(list(row_values)) != len(list(self.__columns))): # if the number of values in the row is greater than the number of columns
            raise ValueError(f'mismatched number of columns and values expecting {len(list(self.__columns))}, found {len(list(row_values))}')
        self.__row_data.append(DataRow(self.__columns, row_values)) # append the row to the table

    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        if not set(row_indexes) <= set(range(len(self.__row_data))): # if row_indexes is not a subset of the set of original rows
            raise IndexError('bad row index')
        table = DataTable(self.columns()) # create new table with same columns
        for index in row_indexes:
            table.append(self[index].values()) # append each row kept to the new table
        return table
    
    def drop(self, columns):
        """Removes the given columns from the current table.
        Args:
        column: the name of the columns to drop
        """

        for i in columns:
            if i in self.__columns:
                self.__columns.remove(i)

        for i in self:
            for j in columns:
                del i[j]

        return self
        
    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        if row_index >= len(self.__row_data): # if row_index is out of bounds
            raise IndexError('bad row index')
        if column not in self.columns(): # if column is not in the list of columns
            raise IndexError('bad column name')
        self.__row_data[row_index][column] = new_value # update the value of the column in the row at row_index

    
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """

        # If index names are repeated in the columns list, raise an IndexError
        if len(columns) != len(set(columns)):
            raise IndexError('Indexes are repeated')
        
        # If columns is not a subset of the columns in both tables, raise an IndexError
        if not set(columns) <= set(table1.columns()) or not set(columns) <= set(table2.columns()):
            raise IndexError('Column indices are not found in both tables')
        
        if columns == []: # if columns is empty, perform the cartesian product
            combined = DataTable(table1.columns() + table2.columns()) # create new table with the given columns
            for row1 in table1: 
                for row2 in table2: 
                    combined.append(row1.values() + row2.values()) # append the combined row to the new table
            return combined

        else:
            cleaned = table2.columns()

            for i in table2.columns():
                if i in columns: # if the column is in the list of columns being combined on
                    cleaned.remove(i) # remove the column from the list of columns

            combined = DataTable(table1.columns() + cleaned) # create a new table with the given columns and remove duplicates

            shorter_t2 = table2.copy() # create a copy of table2

            for row in shorter_t2:
                for i in columns:
                    del row[i]

            for r1 in table1: # r1 is a DataRow object
                count = 0
                for r2 in table2: # r2 is a DataRow object
                    if r1.values(columns) == r2.values(columns): # if the values of the columns being combined on are equal
                        shorter_r2 = shorter_t2[count] # shorter_r2 is at the current itteration of the for loop
                        combined_rows = (r1 + shorter_r2).values()
                        temp = DataRow(r1.columns() + shorter_r2.columns(), combined_rows) # combine the rows
                        combined.append(temp.values()) # append the combined row to the new table
                    count += 1

            if non_matches == False: 
                return combined
            
            else:
                
                combined_len = len(combined.columns())

                t1_diff = [] # t1_diff is the list of names of the columns in combined that are unqiue to table2
                t2_diff = [] # t2_diff is the list of names of the columns in combined that are unqiue to table1

                for i in combined.columns(): 
                    if i not in table1.columns():
                        t1_diff.append(i)
                for j in combined.columns(): 
                    if j not in table2.columns():
                        t2_diff.append(j)

                selected_columns = [] # selected_columns is the list of rows that have already been selected
                for r3 in combined:
                    selected_columns.append(r3.values(columns))

                for r1 in table1:
                    if r1.values(columns) not in selected_columns:
                        temp_row = [""] * combined_len # temp_row is a list of empty strings
                        for i in combined.columns(): # combined.columns() is a list of the column names
                            if i not in t1_diff: # If we are not at an empty string location
                                index1 = combined.columns().index(i) # index is the index of the column name in the list of column names
                                index2 = table1.columns().index(i)
                                temp_row[index1] = r1.values()[index2]
                        temp = DataRow(combined.columns(), temp_row)
                        combined.append(temp.values())

                for r2 in table2:
                    if r2.values(columns) not in selected_columns:
                        temp_row = [""] * combined_len # temp_row is a list of empty strings
                        for i in combined.columns(): # combined.columns() is a list of the column names
                            if i not in t2_diff: # If we are not at an empty string location
                                index1 = combined.columns().index(i) # index is the index of the column name in the list of column names
                                index2 = table2.columns().index(i)
                                temp_row[index1] = r2.values()[index2]
                        temp = DataRow(combined.columns(), temp_row)
                        combined.append(temp.values())

                return combined
    
    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        if isinstance(value, str): # if value is a string
            try:
                return int(value) # try to convert to int
            except ValueError:
                try:
                    return float(value) # try to convert to float
                except ValueError:
                    return value # return the original value
        else:
            return value # return the original value