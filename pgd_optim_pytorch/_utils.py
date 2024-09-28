def filter_df(df, **kwargs):
    # Start with the full DataFrame
    filtered_df = df

    # Apply each condition from kwargs
    for column, value in kwargs.items():
        filtered_df = filtered_df[filtered_df[column] == value]

    return filtered_df
