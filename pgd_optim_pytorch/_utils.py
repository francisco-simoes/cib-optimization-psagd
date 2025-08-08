from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.preprocessing import MinMaxScaler


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


def parallel_coordinates_plot(
    df,
    columns_to_plot: list[str],
    label_renaming: Optional[dict[str, str]] = None,
    y_shift: float = 0.01,
    savefig=True,
    savepath="parallelplot.png",
    start_at_zero: list[str] = [],
):

    # Enable full LaTeX rendering
    plt.rcParams["text.usetex"] = True

    plt.rcParams.update(
        {
            "font.size": 14,  # General font size
            "axes.labelsize": 16,  # Axis label font size
            "axes.titlesize": 18,  # Title font size
            # "xtick.labelsize": 14,  # X-axis tick labels
            "xtick.labelsize": 20,  # X-axis tick labels
            "ytick.labelsize": 14,  # Y-axis tick labels
            "legend.fontsize": 12,  # Legend font size
        }
    )

    # Normalize each column separately for independent scaling
    #
    # For the specific column you want to normalize from zero
    aux_series = {}
    for col in start_at_zero:
        aux_series[col] = df[col] / df[col].max()

    scaler = MinMaxScaler()
    if start_at_zero != []:
        cols_to_normalize = [col for col in columns_to_plot if col not in start_at_zero]
        # Normalize selected columns
        normalized_columns = pd.DataFrame(
            scaler.fit_transform(df[cols_to_normalize]), columns=cols_to_normalize
        )
        # Add back the excluded columns
        for col in start_at_zero:
            normalized_data = pd.concat(
                [
                    normalized_columns.reset_index(drop=True),
                    aux_series[col].reset_index(drop=True),
                ],
                axis=1,
            )

    else:
        normalized_data = pd.DataFrame(
            scaler.fit_transform(df[columns_to_plot]), columns=columns_to_plot
        )
    # Define the number of features (axes)
    num_features = len(columns_to_plot)
    x_positions = np.arange(num_features)  # X positions of axes

    # Convert DataFrame to NumPy array for easier iteration
    data_array = normalized_data.to_numpy()

    # Prepare line segments for LineCollection
    lines = []
    colors = []

    for i in range(len(df)):  # Iterate over rows
        y_values = data_array[i]  # Get normalized values
        points = np.column_stack([x_positions, y_values])  # X and Y pairs
        lines.append(points)

        # Assign color based on 'gamma' value
        colors.append(df["gamma"].iloc[i])

    # Normalize colors for color mapping
    colors = np.array(colors)
    color_norm = (colors - colors.min()) / (colors.max() - colors.min())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use LineCollection to plot multiple lines
    lc = LineCollection(lines, cmap="coolwarm", array=color_norm, alpha=0.8)
    ax.add_collection(lc)

    # Set labels for each axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [label_renaming[label] for label in columns_to_plot], rotation=45
    )

    # Set axis limits
    # ax.set_xlim(-0.5, num_features - 0.5)
    # y_shift = 0.1 # for visibility of lines on 0 and 1
    ax.set_ylim(-y_shift, 1 + y_shift)

    # Add vertical axes for each feature (with its own scale)
    # xticks_positions = ax.get_xticks()
    # print(xticks_positions)
    xticks_positions = ax.transData.transform([(xtick, 0) for xtick in ax.get_xticks()])
    for i, col in enumerate(columns_to_plot):
        # min_val, max_val = df[col].min(), df[col].max()
        min_val, max_val = normalized_data[col].min(), normalized_data[col].max()

        # Create a secondary y-axis for each feature
        ax_sec = ax.twinx()
        ax_sec.set_ylim(min_val - y_shift, max_val + y_shift)
        ax_sec.set_yticks(np.linspace(min_val, max_val, num=6))  # 5 tick marks
        # ax_sec.set_yticklabels(
        #     df[col].sort_values(ascending=True).round(3)
        # )  # Set custom labels
        if col in start_at_zero:
            min_val_unnormalize, max_val_unnormalize = 0.0, df[col].max()
        else:
            min_val_unnormalize, max_val_unnormalize = df[col].min(), df[col].max()
        # Set custom labels
        ax_sec.set_yticklabels(
            [
                f"{val:.3f}"
                for val in np.linspace(min_val_unnormalize, max_val_unnormalize, num=6)
            ]
        )
        ax_sec.yaxis.set_label_position("right")
        ax_sec.yaxis.set_ticks_position("right")
        spine_pos = i / (num_features - 1)
        ax_sec.spines["right"].set_position(("axes", spine_pos))
        # ax_sec.set_ylabel(col, fontsize=10)
        ax_sec.spines["top"].set_visible(False)  # Remove upper horizontal line
        ax_sec.spines["bottom"].set_visible(False)  # Remove lower horizontal line

    ax.spines["top"].set_visible(False)  # Remove upper horizontal line
    ax.spines["bottom"].set_visible(False)  # Remove lower horizontal line
    ax.yaxis.set_visible("False")
    ax.set_yticks([])  # Remove y-axis ticks

    # Add colorbar for gamma
    # cbar = plt.colorbar(lc, ax=ax)
    # cbar.set_label('Gamma')

    # Title and display
    # ax.set_title("")

    plt.subplots_adjust(bottom=0.2)  # Increase space at the bottom

    if savefig:
        plt.savefig(savepath)
    plt.show()
