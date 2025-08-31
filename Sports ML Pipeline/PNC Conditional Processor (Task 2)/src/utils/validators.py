def validate_columns(df, selected_columns):
    """Validate that selected columns exist in the DataFrame."""
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the dataset: {', '.join(missing_columns)}")

def validate_thresholds(thresholds, selected_columns):
    """Validate that thresholds are provided for all selected columns."""
    for col in selected_columns:
        if col not in thresholds:
            raise ValueError(f"No threshold defined for column: {col}")
        if not isinstance(thresholds[col], (int, float)):
            raise ValueError(f"Threshold for column {col} must be a number.")

def validate_data_types(df, selected_columns):
    """Validate that the selected columns have appropriate data types for analysis."""
    for col in selected_columns:
        if df[col].dtype not in [int, float, object]:
            raise ValueError(f"Column {col} has an unsupported data type: {df[col].dtype}.")