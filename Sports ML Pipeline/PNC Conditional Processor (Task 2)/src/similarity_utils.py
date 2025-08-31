import pandas as pd


def add_similarity_columns(df: pd.DataFrame, group_by_cols: list, sum_cols: list) -> pd.DataFrame:
    """
    Adds per-row similarity counts and per-group sums to the DataFrame.

    - similar_count: number of rows that share the same values for all columns in group_by_cols
    - similar_sum_<col>: sum of <col> over the group defined by group_by_cols

    If group_by_cols is empty, returns the original DataFrame.
    """
    if not group_by_cols:
        return df

    df = df.copy()

    try:
        grouped = df.groupby(group_by_cols, dropna=False)
    except Exception:
        return df

    # similar_count
    df["similar_count"] = grouped.transform("size").iloc[:, 0]

    # similar sums for numeric sum_cols
    for col in sum_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"similar_sum_{col}"] = grouped[col].transform("sum")

    return df


if __name__ == "__main__":
    # quick test
    data = {
        'id': [1, 2, 3, 4, 5, 6],
        'groupA': ['x', 'x', 'y', 'x', 'y', 'x'],
        'groupB': [10, 10, 20, 10, 20, 10],
        'value1': [5, 7, 3, 2, 1, 4],
        'value2': [100, 200, 100, 100, 300, 100]
    }
    df = pd.DataFrame(data)
    print("Original:\n", df)
    out = add_similarity_columns(df, ['groupA', 'groupB'], ['value1', 'value2'])
    print("\nResult:\n", out)
