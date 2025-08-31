import pandas as pd
def apply_filters(df, selected_columns, thresholds):
    filtered_df = df.copy()
    
    for col in selected_columns:
        if col in filtered_df.columns:
            threshold = thresholds.get(col)
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                filtered_df = filtered_df[filtered_df[col] > threshold]
            else:
                mode_value = filtered_df[col].mode().iloc[0]
                filtered_df = filtered_df[filtered_df[col] == mode_value]

    return filtered_df


def get_remaining_columns(original_df, filtered_df):
    remaining_columns = original_df.columns.difference(filtered_df.columns)
    return remaining_columns.tolist()