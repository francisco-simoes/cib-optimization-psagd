def filter_df(df, **kwargs):
    """Filter a DataFrame based on specified conditions.

    Each keyword argument represents a column
    name and the value to filter for that column. The function returns a new
    DataFrame containing only the rows that match all specified conditions.

    Example:
    ---------
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3], 'B': ['x', 'y', 'z']}
    >>> df = pd.DataFrame(data)
    >>> filtered = filter_df(df, A=2, B='y')
    >>> print(filtered)
       A  B
    1  2  y
    """
    filtered_df = df

    # Apply each condition from kwargs
    for column, value in kwargs.items():
        filtered_df = filtered_df[filtered_df[column] == value]

    return filtered_df
