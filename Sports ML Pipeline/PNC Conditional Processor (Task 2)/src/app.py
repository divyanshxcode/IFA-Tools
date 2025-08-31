import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
from datetime import datetime, date
from data_processor import load_and_process_data
from analysis_engine import analyze_data_combinations, is_date_column, get_date_columns
from excel_handler import export_results
from similarity_utils import add_similarity_columns

def parse_date_column(df: pd.DataFrame, column_name: str, date_format: str) -> tuple[pd.DataFrame, bool, str]:
    """
    Parse a date column with the specified format and convert to datetime.
    
    Returns:
        tuple: (modified_dataframe, success_flag, error_message)
    """
    try:
        df_copy = df.copy()
        
        # Handle different format specifications
        if date_format == "auto":
            # Try pandas automatic detection with errors='coerce'
            df_copy[column_name] = pd.to_datetime(df_copy[column_name], errors='coerce')
        elif date_format == "excel_serial":
            # Handle Excel serial date numbers
            df_copy[column_name] = pd.to_datetime(df_copy[column_name], origin='1899-12-30', unit='D', errors='coerce')
        else:
            # Use specific format with errors='coerce'
            df_copy[column_name] = pd.to_datetime(df_copy[column_name], format=date_format, errors='coerce')
        
        # Check for any NaT values after conversion
        nat_count = df_copy[column_name].isna().sum()
        total_count = len(df_copy)
        original_na_count = df[column_name].isna().sum()
        
        # Calculate actual parsing failures (excluding original NAs)
        parsing_failures = nat_count - original_na_count
        
        if nat_count == total_count:
            return df, False, "All values failed to parse. Please check the date format."
        elif parsing_failures > 0:
            success_rate = ((total_count - parsing_failures) / total_count) * 100
            return df_copy, True, f"Parsed successfully with {success_rate:.1f}% success rate ({parsing_failures} values failed to parse)"
        else:
            return df_copy, True, "All dates parsed successfully!"
            
    except Exception as e:
        return df, False, f"Parsing failed: {str(e)}"

def parse_date_column_multi_format(df: pd.DataFrame, column_name: str, date_formats: list) -> tuple[pd.DataFrame, bool, str]:
    """
    Parse a date column with multiple possible formats and convert to datetime.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to parse
        date_formats: List of format strings to try
    
    Returns:
        tuple: (modified_dataframe, success_flag, detailed_message)
    """
    try:
        df_copy = df.copy()
        original_values = df_copy[column_name].copy()
        original_na_count = original_values.isna().sum()
        
        # Initialize with NaT
        parsed_values = pd.Series([pd.NaT] * len(df_copy), index=df_copy.index)
        format_success_counts = {}
        
        # Get non-null values to work with
        non_null_mask = ~original_values.isna()
        values_to_parse = original_values[non_null_mask]
        
        if len(values_to_parse) == 0:
            return df, False, "No non-null values to parse"
        
        # Try each format
        for date_format in date_formats:
            # Skip values that are already successfully parsed
            remaining_mask = non_null_mask & parsed_values.isna()
            remaining_values = original_values[remaining_mask]
            
            if len(remaining_values) == 0:
                continue
                
            format_success_counts[date_format] = 0
            
            if date_format == "auto":
                # Try pandas automatic detection
                try:
                    temp_parsed = pd.to_datetime(remaining_values, errors='coerce')
                    success_mask = ~temp_parsed.isna()
                    parsed_values.loc[remaining_mask & success_mask] = temp_parsed[success_mask]
                    format_success_counts[date_format] = success_mask.sum()
                except:
                    continue
                    
            elif date_format == "excel_serial":
                # Handle Excel serial date numbers
                try:
                    # Only try this for numeric values
                    numeric_mask = pd.to_numeric(remaining_values, errors='coerce').notna()
                    if numeric_mask.any():
                        numeric_values = remaining_values[numeric_mask]
                        temp_parsed = pd.to_datetime(numeric_values, origin='1899-12-30', unit='D', errors='coerce')
                        success_mask = ~temp_parsed.isna()
                        
                        # Map back to original indices
                        remaining_numeric_indices = remaining_values[numeric_mask].index
                        parsed_values.loc[remaining_numeric_indices[success_mask]] = temp_parsed[success_mask]
                        format_success_counts[date_format] = success_mask.sum()
                except:
                    continue
                    
            else:
                # Use specific format
                try:
                    temp_parsed = pd.to_datetime(remaining_values, format=date_format, errors='coerce')
                    success_mask = ~temp_parsed.isna()
                    parsed_values.loc[remaining_mask & success_mask] = temp_parsed[success_mask]
                    format_success_counts[date_format] = success_mask.sum()
                except:
                    continue
        
        # Update the dataframe
        df_copy[column_name] = parsed_values
        
        # Calculate success statistics
        total_count = len(df_copy)
        final_na_count = parsed_values.isna().sum()
        parsing_failures = final_na_count - original_na_count
        successfully_parsed = total_count - final_na_count
        
        if successfully_parsed == 0:
            return df, False, "All values failed to parse with all attempted formats."
        
        # Generate detailed message
        success_rate = (successfully_parsed / (total_count - original_na_count)) * 100 if (total_count - original_na_count) > 0 else 0
        
        message_parts = [f"Successfully parsed {successfully_parsed}/{total_count - original_na_count} values ({success_rate:.1f}% success rate)"]
        
        # Add format breakdown
        successful_formats = {fmt: count for fmt, count in format_success_counts.items() if count > 0}
        if successful_formats:
            message_parts.append("Format breakdown:")
            for fmt, count in successful_formats.items():
                format_name = fmt if fmt in ["auto", "excel_serial"] else f"Custom: {fmt}"
                message_parts.append(f"  - {format_name}: {count} values")
        
        if parsing_failures > 0:
            message_parts.append(f"{parsing_failures} values failed to parse")
            
        return df_copy, True, "\n".join(message_parts)
        
    except Exception as e:
        return df, False, f"Parsing failed with error: {str(e)}"

def get_auto_detected_date_formats(sample_values):
    """
    Automatically detect potential date formats from sample values
    """
    formats_to_try = [
        "auto",  # pandas auto-detection first
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%d/%m/yy",
        "%m/%d/yy",
        "%y/%m/%d",
        "%d-%m-yy",
        "%m-%d-yy",
        "%y-%m-%d",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "excel_serial"
    ]
    
    # Analyze sample values to suggest most likely formats
    sample_strings = [str(val) for val in sample_values if pd.notna(val)]
    
    suggested_formats = []
    format_priority = {}
    
    for sample in sample_strings[:20]:  # Check first 20 samples
        sample_str = str(sample).strip()
        
        # Check for common patterns
        if len(sample_str) == 10 and sample_str.count('-') == 2:
            format_priority["%Y-%m-%d"] = format_priority.get("%Y-%m-%d", 0) + 1
        elif len(sample_str) == 19 and ' ' in sample_str and ':' in sample_str:
            format_priority["%Y-%m-%d %H:%M:%S"] = format_priority.get("%Y-%m-%d %H:%M:%S", 0) + 1
        elif sample_str.count('/') == 2:
            if len(sample_str.split('/')[0]) == 4:
                format_priority["%Y/%m/%d"] = format_priority.get("%Y/%m/%d", 0) + 1
            elif len(sample_str.split('/')[2]) == 4:
                format_priority["%d/%m/%Y"] = format_priority.get("%d/%m/%Y", 0) + 1
                format_priority["%m/%d/%Y"] = format_priority.get("%m/%d/%Y", 0) + 1
        elif sample_str.replace('.', '').replace('-', '').isdigit():
            # Could be excel serial number
            try:
                num_val = float(sample_str)
                if 1 <= num_val <= 100000:  # Reasonable range for excel serial dates
                    format_priority["excel_serial"] = format_priority.get("excel_serial", 0) + 1
            except:
                pass
    
    # Sort by priority and add to suggested formats
    sorted_formats = sorted(format_priority.items(), key=lambda x: x[1], reverse=True)
    suggested_formats = [fmt for fmt, _ in sorted_formats]
    
    # Always include auto detection first
    if "auto" not in suggested_formats:
        suggested_formats.insert(0, "auto")
    
    # Add remaining common formats
    for fmt in formats_to_try:
        if fmt not in suggested_formats:
            suggested_formats.append(fmt)
    
    return suggested_formats

def get_common_date_formats():
    """Return common date format patterns"""
    return {
        "Auto Detection": "auto",
        "yyyy-mm-dd": "%Y-%m-%d",
        "dd/mm/yyyy": "%d/%m/%Y",
        "mm/dd/yyyy": "%m/%d/%Y", 
        "yyyy/mm/dd": "%Y/%m/%d",
        "dd-mm-yyyy": "%d-%m-%Y",
        "mm-dd-yyyy": "%m-%d-%Y",
        "dd.mm.yyyy": "%d.%m.%Y",
        "mm.dd.yyyy": "%m.%d.%Y",
        "yyyy.mm.dd": "%Y.%m.%d",
        "dd/mm/yy": "%d/%m/%y",
        "mm/dd/yy": "%m/%d/%y",
        "yy/mm/dd": "%y/%m/%d",
        "dd-mm-yy": "%d-%m-%y",
        "mm-dd-yy": "%m-%d-%y",
        "yy-mm-dd": "%y-%m-%d",
        "yyyy-mm-dd hh:mm:ss": "%Y-%m-%d %H:%M:%S",
        "dd/mm/yyyy hh:mm:ss": "%d/%m/%Y %H:%M:%S",
        "mm/dd/yyyy hh:mm:ss": "%m/%d/%Y %H:%M:%S",
        "Excel Serial Number": "excel_serial"
    }

def is_column_datetime_converted(df: pd.DataFrame, column_name: str) -> bool:
    """Check if a column has been successfully converted to datetime"""
    return pd.api.types.is_datetime64_any_dtype(df[column_name])

def filter_dataframe_by_date(df: pd.DataFrame, column: str, filter_config: dict) -> pd.DataFrame:
    """
    Filter dataframe based on date conditions.
    
    Args:
        df: DataFrame to filter
        column: Date column name
        filter_config: Dictionary containing filter type and parameters
    
    Returns:
        Filtered DataFrame
    """
    df_copy = df.copy()
    col_data = pd.to_datetime(df_copy[column])
    
    if filter_config["type"] == "range":
        start_date = pd.to_datetime(filter_config["start_date"])
        end_date = pd.to_datetime(filter_config["end_date"])
        mask = (col_data >= start_date) & (col_data <= end_date)
    elif filter_config["type"] == "before":
        target_date = pd.to_datetime(filter_config["date"])
        mask = col_data < target_date
    elif filter_config["type"] == "after":
        target_date = pd.to_datetime(filter_config["date"])
        mask = col_data > target_date
    elif filter_config["type"] == "on":
        target_date = pd.to_datetime(filter_config["date"])
        mask = col_data.dt.date == target_date
    else:
        return df_copy
    
    return df_copy[mask]

def calculate_consistency(series: pd.Series) -> float:
    """
    Calculates consistency as the percentage of the most frequent value 
    compared to the total count (works for numeric or categorical data).
    """
    if series.empty:
        return 0.0
    value_counts = series.value_counts(normalize=True)
    return float(value_counts.iloc[0] * 100)  # percentage of most common value

def calculate_max_run(series: pd.Series) -> int:
    """
    Calculates the longest consecutive run (streak) of the same value in the series.
    Works for both numeric and categorical data.
    """
    if series.empty:
        return 0
    
    max_run = current_run = 1
    prev_value = series.iloc[0]

    for value in series.iloc[1:]:
        if value == prev_value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
        prev_value = value

    return max_run

def add_similarity_columns(df: pd.DataFrame, group_by_cols: list, sum_cols: list) -> pd.DataFrame:
    """
    Adds per-row similarity counts and per-group sums to the DataFrame.

    - similar_count: number of rows that share the same values for all columns in group_by_cols
    - similar_sum_<col>: sum of <col> over the group defined by group_by_cols

    If group_by_cols is empty, returns the original DataFrame.
    """
    if not group_by_cols:
        return df

    # work on a copy to avoid mutating original unintentionally
    df = df.copy()

    # Choose a stable column to count; using any column works since group size is same
    try:
        grouped = df.groupby(group_by_cols, dropna=False)
    except Exception:
        # fallback: if grouping fails (e.g., unhashable types), return original
        return df

    # similar_count
    df["similar_count"] = grouped.transform("size").iloc[:, 0]

    # similar sums for numeric sum_cols
    for col in sum_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"similar_sum_{col}"] = grouped[col].transform("sum")

    return df

st.set_page_config(
    page_title="Data Analysis Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue/white minimal theme
st.markdown("""
<style>
    .main-header {
        color: #2563eb;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 0.5rem;
    }
    .section-header {
        color: #1d4ed8;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #60a5fa;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.25rem;
    }
    .stButton > button {
        background-color: #ffffff;
        color: #1f2937;
        border: 2px solid #d1d5db;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #f3f4f6;
        border-color: #9ca3af;
        color: #111827;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: 2px solid #2563eb;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }
    .stButton > button[kind="secondary"] {
        background-color: #6b7280;
        color: white;
        border: 2px solid #6b7280;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #4b5563;
        border-color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Data Analysis Tool</h1>', unsafe_allow_html=True)

# Upload data
st.markdown('<h2 class="section-header">Data Upload</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Select Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Initialize session state for date parsing
    if 'parsed_df' not in st.session_state:
        st.session_state.parsed_df = None
    if 'date_columns_config' not in st.session_state:
        st.session_state.date_columns_config = {}
    
    # Load initial data
    if st.session_state.parsed_df is None:
        df = load_and_process_data(uploaded_file)
        st.session_state.original_df = df.copy()
        st.session_state.parsed_df = df.copy()
    else:
        df = st.session_state.parsed_df.copy()
    
    st.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Date Column Configuration
    st.markdown('<h2 class="section-header">Date Column Configuration</h2>', unsafe_allow_html=True)
    
    configure_dates = st.checkbox("Configure Date Columns")
    
    if configure_dates:
        with st.expander("Date Column Configuration", expanded=True):
            # Select date column
            date_column = st.selectbox(
                "Select the date column to configure",
                options=[None] + df.columns.tolist()
            )
            
            if date_column:
                # Show sample data
                st.write("**Sample data from selected column:**")
                sample_data = df[date_column].dropna().head(10)
                st.write(sample_data.tolist())
                
                # Auto-detect potential formats
                suggested_formats = get_auto_detected_date_formats(sample_data)
                
                # Date format selection with auto-suggestions
                st.write("**Parsing Strategy:**")
                parsing_strategy = st.radio(
                    "Choose parsing approach",
                    ["Auto-detect multiple formats", "Specify single format", "Try multiple specific formats"],
                    key=f"parsing_strategy_{date_column}"
                )
                
                if parsing_strategy == "Auto-detect multiple formats":
                    # Use the top suggested formats automatically
                    selected_formats = suggested_formats[:8]  # Use top 8 formats
                    
                    st.info(f"Will try these formats in order: {', '.join(selected_formats[:5])}{'...' if len(selected_formats) > 5 else ''}")
                    
                    # Test parsing button
                    if st.button("Test Multi-Format Parsing", type="secondary"):
                        with st.spinner("Testing multi-format date parsing..."):
                            test_df, success, message = parse_date_column_multi_format(df, date_column, selected_formats)
                            
                            if success:
                                st.success("Multi-format parsing successful!")
                                st.text(message)
                                
                                # Show before/after comparison
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Original Values (first 10):**")
                                    original_sample = df[date_column].head(10)
                                    for i, val in enumerate(original_sample):
                                        st.write(f"{i+1}. {val}")
                                with col2:
                                    st.write("**Parsed Values (first 10):**")
                                    parsed_sample = test_df[date_column].head(10)
                                    for i, val in enumerate(parsed_sample):
                                        if pd.notna(val):
                                            st.write(f"{i+1}. {val.strftime('%Y-%m-%d %H:%M:%S')}")
                                        else:
                                            st.write(f"{i+1}. Failed to parse")
                                
                                # Store test result
                                st.session_state.test_result = {
                                    'column': date_column,
                                    'formats': selected_formats,
                                    'parsed_df': test_df,
                                    'success': success,
                                    'message': message,
                                    'type': 'multi_format'
                                }
                                
                            else:
                                st.error(message)
                                if 'test_result' in st.session_state:
                                    del st.session_state.test_result
                
                elif parsing_strategy == "Specify single format":
                    # Original single format approach
                    date_formats = get_common_date_formats()
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        format_choice = st.selectbox(
                            "Select date format",
                            options=list(date_formats.keys()),
                            index=0
                        )
                    
                    with col2:
                        if format_choice not in ["Auto Detection", "Excel Serial Number"]:
                            custom_format = st.text_input(
                                "Or enter custom format",
                                placeholder="e.g., %d/%m/%Y %H:%M:%S",
                                help="Use Python strftime format codes"
                            )
                            if custom_format.strip():
                                selected_format = custom_format.strip()
                            else:
                                selected_format = date_formats[format_choice]
                        else:
                            selected_format = date_formats[format_choice]
                            st.info(f"Using: {format_choice}")
                    
                    # Test parsing button
                    if st.button("Test Single Format Parsing", type="secondary"):
                        with st.spinner("Testing date parsing..."):
                            test_df, success, message = parse_date_column(df, date_column, selected_format)
                            
                            if success:
                                st.success(message)
                                
                                # Show before/after comparison
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Original Values (first 5):**")
                                    st.write(df[date_column].head(5).tolist())
                                with col2:
                                    st.write("**Parsed Values (first 5):**")
                                    parsed_values = test_df[date_column].head(5)
                                    st.write([d.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(d) else 'Failed to parse' for d in parsed_values])
                                
                                st.session_state.test_result = {
                                    'column': date_column,
                                    'format': selected_format,
                                    'parsed_df': test_df,
                                    'success': success,
                                    'message': message,
                                    'type': 'single_format'
                                }
                                
                            else:
                                st.error(message)
                                st.info("Try multi-format parsing or a different format")
                                if 'test_result' in st.session_state:
                                    del st.session_state.test_result
                
                else:  # Try multiple specific formats
                    st.write("**Select multiple formats to try:**")
                    date_formats = get_common_date_formats()
                    
                    # Pre-select suggested formats
                    default_selections = []
                    for fmt_name, fmt_code in date_formats.items():
                        if fmt_code in suggested_formats[:6]:
                            default_selections.append(fmt_name)
                    
                    selected_format_names = st.multiselect(
                        "Choose formats to try (in order)",
                        options=list(date_formats.keys()),
                        default=default_selections
                    )
                    
                    if selected_format_names:
                        selected_formats = [date_formats[name] for name in selected_format_names]
                        
                        # Custom format option
                        custom_formats_text = st.text_area(
                            "Additional custom formats (one per line)",
                            placeholder="%d-%b-%Y\n%Y%m%d\n%d.%m.%Y %H:%M",
                            help="Enter additional Python strftime format codes"
                        )
                        
                        if custom_formats_text.strip():
                            custom_formats = [fmt.strip() for fmt in custom_formats_text.split('\n') if fmt.strip()]
                            selected_formats.extend(custom_formats)
                        
                        st.info(f"Will try {len(selected_formats)} formats: {', '.join(selected_formats[:3])}{'...' if len(selected_formats) > 3 else ''}")
                        
                        # Test parsing button
                        if st.button("Test Multiple Format Parsing", type="secondary"):
                            with st.spinner("Testing multi-format date parsing..."):
                                test_df, success, message = parse_date_column_multi_format(df, date_column, selected_formats)
                                
                                if success:
                                    st.success("Multi-format parsing successful!")
                                    st.text(message)
                                    
                                    # Show before/after comparison
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Original Values (first 10):**")
                                        original_sample = df[date_column].head(10)
                                        for i, val in enumerate(original_sample):
                                            st.write(f"{i+1}. {val}")
                                    with col2:
                                        st.write("**Parsed Values (first 10):**")
                                        parsed_sample = test_df[date_column].head(10)
                                        for i, val in enumerate(parsed_sample):
                                            if pd.notna(val):
                                                st.write(f"{i+1}. {val.strftime('%Y-%m-%d %H:%M:%S')}")
                                            else:
                                                st.write(f"{i+1}. Failed to parse")
                                    
                                    st.session_state.test_result = {
                                        'column': date_column,
                                        'formats': selected_formats,
                                        'parsed_df': test_df,
                                        'success': success,
                                        'message': message,
                                        'type': 'multi_format'
                                    }
                                    
                                else:
                                    st.error(message)
                                    if 'test_result' in st.session_state:
                                        del st.session_state.test_result
                
                # Apply parsing if test was successful
                if 'test_result' in st.session_state and st.session_state.test_result['column'] == date_column:
                    if st.button("Apply Date Parsing", type="primary"):
                        # Apply the parsing to the main dataframe in session state
                        st.session_state.parsed_df = st.session_state.test_result['parsed_df'].copy()
                        
                        # Store the configuration
                        if st.session_state.test_result['type'] == 'multi_format':
                            st.session_state.date_columns_config[date_column] = {
                                'formats': st.session_state.test_result['formats'],
                                'applied': True,
                                'type': 'multi_format'
                            }
                        else:
                            st.session_state.date_columns_config[date_column] = {
                                'format': st.session_state.test_result['format'],
                                'applied': True,
                                'type': 'single_format'
                            }
                        
                        st.success(f"Date parsing applied successfully to column '{date_column}'!")
                        
                        # Clear the test result
                        del st.session_state.test_result
                        
                        # Rerun to refresh the dataframe
                        st.rerun()
    
    # Select columns for analysis
    st.markdown('<h2 class="section-header">Column Selection</h2>', unsafe_allow_html=True)
    
    # Show column types for user reference
    with st.expander("Column Type Reference"):
        # Get date columns (both originally detected and manually configured)
        date_cols = get_date_columns(df)
        for col, config in st.session_state.date_columns_config.items():
            if config.get('applied', False) and col not in date_cols:
                date_cols.append(col)
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in date_cols and col not in numeric_cols]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Date Columns:**")
            if date_cols:
                for col in date_cols:
                    configured_mark = " ✓" if col in st.session_state.date_columns_config else ""
                    st.write(f"• {col}{configured_mark}")
            else:
                st.write("None detected")
                
        with col2:
            st.write("**Numeric Columns:**")
            if numeric_cols:
                for col in numeric_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(numeric_cols) > 10:
                    st.write(f"... and {len(numeric_cols) - 10} more")
            else:
                st.write("None detected")
                
        with col3:
            st.write("**Categorical Columns:**")
            if categorical_cols:
                for col in categorical_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(categorical_cols) > 10:
                    st.write(f"... and {len(categorical_cols) - 10} more")
            else:
                st.write("None detected")
    
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns for analysis", columns)

    if selected_columns:
        st.markdown('<h2 class="section-header">Statistical Analysis & Threshold Configuration</h2>', unsafe_allow_html=True)
        thresholds = {}
        
        for col in selected_columns:
            with st.expander(f"Configure {col}"):
                col_data = df[col].dropna()
                
                # Check if column is date/datetime - improved detection
                is_date_col = (
                    is_date_column(df, col) or 
                    is_column_datetime_converted(df, col) or
                    (col in st.session_state.date_columns_config and 
                     st.session_state.date_columns_config[col].get('applied', False))
                )
                
                if is_date_col:
                    # Ensure the column is properly converted to datetime
                    try:
                        if not is_column_datetime_converted(df, col):
                            # If not already datetime, try to convert
                            col_data_dt = pd.to_datetime(col_data, errors='coerce')
                        else:
                            col_data_dt = col_data
                    except:
                        col_data_dt = col_data
                    
                    # Show statistics for date columns
                    valid_dates = col_data_dt.dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        date_range = max_date - min_date
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Earliest Date", min_date.strftime('%Y-%m-%d'))
                        with col2:
                            st.metric("Latest Date", max_date.strftime('%Y-%m-%d'))
                        with col3:
                            st.metric("Date Range", f"{date_range.days} days")
                        with col4:
                            st.metric("Valid Dates", f"{len(valid_dates)}/{len(col_data)}")
                        
                        # Date filtering method selection
                        date_method = st.selectbox(
                            f"Date filtering method for {col}",
                            ["Single Range", "Multiple Ranges", "Before Dates", "After Dates", "On Dates", "Last N Days", "First N Days"],
                            key=f"date_method_{col}"
                        )
                        
                        if date_method == "Single Range":
                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input(
                                    f"Start date for {col}",
                                    value=min_date.date(),
                                    min_value=min_date.date(),
                                    max_value=max_date.date(),
                                    key=f"single_start_date_{col}"
                                )
                            with col2:
                                end_date = st.date_input(
                                    f"End date for {col}",
                                    value=max_date.date(),
                                    min_value=min_date.date(),
                                    max_value=max_date.date(),
                                    key=f"single_end_date_{col}"
                                )
                            
                            if start_date <= end_date:
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[(df[col].dt.date >= start_date) & 
                                                               (df[col].dt.date <= end_date)])
                                    else:
                                        filtered_count = len(df[(pd.to_datetime(df[col], errors='coerce').dt.date >= start_date) & 
                                                               (pd.to_datetime(df[col], errors='coerce').dt.date <= end_date)])
                                    st.info(f"Range: {start_date} to {end_date} ({filtered_count} records)")
                                except:
                                    st.info(f"Range: {start_date} to {end_date}")
                                
                                thresholds[col] = {
                                    "type": "single_range",
                                    "start_date": start_date,
                                    "end_date": end_date
                                }
                            else:
                                st.error("Start date must be before or equal to end date")
                                
                        elif date_method == "Multiple Ranges":
                            num_ranges = st.number_input(
                                f"Number of date ranges for {col}",
                                min_value=1,
                                max_value=20,
                                value=3,
                                key=f"num_ranges_{col}"
                            )
                            
                            ranges = []
                            for i in range(num_ranges):
                                st.write(f"**Range {i+1}:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    start_date = st.date_input(
                                        f"Start date",
                                        value=min_date.date(),
                                        min_value=min_date.date(),
                                        max_value=max_date.date(),
                                        key=f"multi_range_start_{col}_{i}"
                                    )
                                with col2:
                                    end_date = st.date_input(
                                        f"End date",
                                        value=max_date.date(),
                                        min_value=min_date.date(),
                                        max_value=max_date.date(),
                                        key=f"multi_range_end_{col}_{i}"
                                    )
                                
                                if start_date <= end_date:
                                    try:
                                        if is_column_datetime_converted(df, col):
                                            filtered_count = len(df[(df[col].dt.date >= start_date) & 
                                                                   (df[col].dt.date <= end_date)])
                                        else:
                                            filtered_count = len(df[(pd.to_datetime(df[col], errors='coerce').dt.date >= start_date) & 
                                                                   (pd.to_datetime(df[col], errors='coerce').dt.date <= end_date)])
                                        st.info(f"{start_date} to {end_date} ({filtered_count} records)")
                                    except:
                                        st.info(f"{start_date} to {end_date}")
                                    ranges.append({"start_date": start_date, "end_date": end_date})
                                else:
                                    st.error(f"Range {i+1}: Start date must be before or equal to end date")
                            
                            if ranges:
                                thresholds[col] = {
                                    "type": "multiple_ranges",
                                    "ranges": ranges
                                }
                                
                        elif date_method == "Before Dates":
                            num_dates = st.number_input(
                                f"Number of 'before' dates for {col}",
                                min_value=1,
                                max_value=20,
                                value=3,
                                key=f"num_before_dates_{col}"
                            )
                            
                            before_dates = []
                            for i in range(num_dates):
                                selected_date = st.date_input(
                                    f"Before date {i+1}",
                                    value=min_date.date() + pd.Timedelta(days=(i+1) * date_range.days // (num_dates + 1)),
                                    min_value=min_date.date(),
                                    max_value=max_date.date(),
                                    key=f"before_date_{col}_{i}"
                                )
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[df[col].dt.date < selected_date])
                                    else:
                                        filtered_count = len(df[pd.to_datetime(df[col], errors='coerce').dt.date < selected_date])
                                    st.info(f"Before {selected_date} ({filtered_count} records)")
                                except:
                                    st.info(f"Before {selected_date}")
                                before_dates.append(selected_date)
                            
                            thresholds[col] = {
                                "type": "multiple_before",
                                "dates": before_dates
                            }
                            
                        elif date_method == "After Dates":
                            num_dates = st.number_input(
                                f"Number of 'after' dates for {col}",
                                min_value=1,
                                max_value=20,
                                value=3,
                                key=f"num_after_dates_{col}"
                            )
                            
                            after_dates = []
                            for i in range(num_dates):
                                selected_date = st.date_input(
                                    f"After date {i+1}",
                                    value=min_date.date() + pd.Timedelta(days=(i+1) * date_range.days // (num_dates + 1)),
                                    min_value=min_date.date(),
                                    max_value=max_date.date(),
                                    key=f"after_date_{col}_{i}"
                                )
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[df[col].dt.date > selected_date])
                                    else:
                                        filtered_count = len(df[pd.to_datetime(df[col], errors='coerce').dt.date > selected_date])
                                    st.info(f"After {selected_date} ({filtered_count} records)")
                                except:
                                    st.info(f"After {selected_date}")
                                after_dates.append(selected_date)
                            
                            thresholds[col] = {
                                "type": "multiple_after", 
                                "dates": after_dates
                            }
                            
                        elif date_method == "On Dates":
                            num_dates = st.number_input(
                                f"Number of specific dates for {col}",
                                min_value=1,
                                max_value=20,
                                value=3,
                                key=f"num_on_dates_{col}"
                            )
                            
                            on_dates = []
                            for i in range(num_dates):
                                selected_date = st.date_input(
                                    f"On date {i+1}",
                                    value=min_date.date() + pd.Timedelta(days=(i+1) * date_range.days // (num_dates + 1)),
                                    min_value=min_date.date(),
                                    max_value=max_date.date(),
                                    key=f"on_date_{col}_{i}"
                                )
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[df[col].dt.date == selected_date])
                                    else:
                                        filtered_count = len(df[pd.to_datetime(df[col], errors='coerce').dt.date == selected_date])
                                    st.info(f"On {selected_date} ({filtered_count} records)")
                                except:
                                    st.info(f"On {selected_date}")
                                on_dates.append(selected_date)
                            
                            thresholds[col] = {
                                "type": "multiple_on",
                                "dates": on_dates
                            }
                            
                        elif date_method == "Last N Days":
                            num_periods = st.number_input(
                                f"Number of different 'last N days' periods for {col}",
                                min_value=1,
                                max_value=10,
                                value=3,
                                key=f"num_last_periods_{col}"
                            )
                            
                            last_n_configs = []
                            for i in range(num_periods):
                                n_days = st.number_input(
                                    f"Last N days period {i+1}",
                                    min_value=1,
                                    max_value=date_range.days,
                                    value=min((i+1) * 30, date_range.days),
                                    key=f"last_n_days_{col}_{i}"
                                )
                                cutoff_date = max_date - pd.Timedelta(days=n_days)
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[df[col] >= cutoff_date])
                                    else:
                                        filtered_count = len(df[pd.to_datetime(df[col], errors='coerce') >= cutoff_date])
                                    st.info(f"Last {n_days} days (from {cutoff_date.strftime('%Y-%m-%d')}) - {filtered_count} records")
                                except:
                                    st.info(f"Last {n_days} days (from {cutoff_date.strftime('%Y-%m-%d')})")
                                
                                last_n_configs.append({"days": n_days, "cutoff_date": cutoff_date})
                            
                            thresholds[col] = {
                                "type": "multiple_last_n_days",
                                "configs": last_n_configs
                            }
                            
                        elif date_method == "First N Days":
                            num_periods = st.number_input(
                                f"Number of different 'first N days' periods for {col}",
                                min_value=1,
                                max_value=10,
                                value=3,
                                key=f"num_first_periods_{col}"
                            )
                            
                            first_n_configs = []
                            for i in range(num_periods):
                                n_days = st.number_input(
                                    f"First N days period {i+1}",
                                    min_value=1,
                                    max_value=date_range.days,
                                    value=min((i+1) * 30, date_range.days),
                                    key=f"first_n_days_{col}_{i}"
                                )
                                cutoff_date = min_date + pd.Timedelta(days=n_days)
                                try:
                                    if is_column_datetime_converted(df, col):
                                        filtered_count = len(df[df[col] <= cutoff_date])
                                    else:
                                        filtered_count = len(df[pd.to_datetime(df[col], errors='coerce') <= cutoff_date])
                                    st.info(f"First {n_days} days (until {cutoff_date.strftime('%Y-%m-%d')}) - {filtered_count} records")
                                except:
                                    st.info(f"First {n_days} days (until {cutoff_date.strftime('%Y-%m-%d')})")
                                
                                first_n_configs.append({"days": n_days, "cutoff_date": cutoff_date})
                            
                            thresholds[col] = {
                                "type": "multiple_first_n_days",
                                "configs": first_n_configs
                            }
                    else:
                        st.warning(f"No valid dates found in column '{col}'. Please check date parsing configuration.")
                
                # Check if column is numeric
                elif pd.api.types.is_numeric_dtype(col_data):
                    # Show statistics for numeric columns
                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                    with col1:
                        st.metric("Min", f"{col_data.min():.2f}")
                    with col2:
                        st.metric("Max", f"{col_data.max():.2f}")
                    with col3:
                        st.metric("Mean", f"{col_data.mean():.2f}")
                    with col4:
                        st.metric("Median", f"{col_data.median():.2f}")
                    with col5:
                        st.metric("Count", len(col_data))
                    with col6:
                        st.metric("Variance", f"{col_data.var():.2f}")
                    with col7:
                        st.metric("Consistency", f"{calculate_consistency(col_data):.2f}")
                    with col8:
                        st.metric("Max Run", calculate_max_run(col_data))

                    # Threshold selection
                    threshold_type = st.selectbox(
                        f"Threshold method for {col}",
                        ["Mean", "Median", "Custom", "Multiple Conditions (OR Logic)", "Range"],
                        key=f"threshold_type_{col}"
                    )
                    
                    if threshold_type == "Mean":
                        threshold_value = col_data.mean()
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "mean", "value": threshold_value}
                    elif threshold_type == "Median":
                        threshold_value = col_data.median()
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "median", "value": threshold_value}
                    elif threshold_type == "Custom":
                        threshold_value = st.number_input(
                            f"Custom threshold for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"custom_threshold_{col}"
                        )
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "custom", "value": threshold_value}
                    elif threshold_type == "Multiple Conditions (OR Logic)":
                        st.write("**Add multiple conditions (OR logic):**")
                        
                        num_conditions = st.number_input(
                            f"Number of conditions for {col}",
                            min_value=1,
                            max_value=20,
                            value=2,
                            key=f"num_conditions_{col}"
                        )
                        
                        conditions = []
                        for i in range(num_conditions):
                            st.write(f"**Condition {i+1}:**")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                operator = st.selectbox(
                                    f"Operator",
                                    ["Greater than (>)", "Less than (<)", "Greater than or equal (>=)", "Less than or equal (<=)"],
                                    key=f"operator_{col}_{i}"
                                )
                            
                            with col2:
                                value = st.number_input(
                                    f"Value",
                                    min_value=float(col_data.min()),
                                    max_value=float(col_data.max()),
                                    value=float(col_data.mean()),
                                    key=f"condition_value_{col}_{i}"
                                )
                            
                            # Convert operator to symbol
                            operator_map = {
                                "Greater than (>)": ">",
                                "Less than (<)": "<", 
                                "Greater than or equal (>=)": ">=",
                                "Less than or equal (<=)": "<="
                            }
                            
                            conditions.append({
                                "operator": operator_map[operator],
                                "value": value
                            })
                            
                            # Show preview
                            try:
                                if operator_map[operator] == ">":
                                    preview_count = len(col_data[col_data > value])
                                elif operator_map[operator] == "<":
                                    preview_count = len(col_data[col_data < value])
                                elif operator_map[operator] == ">=":
                                    preview_count = len(col_data[col_data >= value])
                                elif operator_map[operator] == "<=":
                                    preview_count = len(col_data[col_data <= value])
                                
                                st.info(f"{col} {operator_map[operator]} {value:.2f} → {preview_count} records")
                            except:
                                pass
                        
                        # Show combined preview
                        try:
                            combined_mask = pd.Series([False] * len(col_data), index=col_data.index)
                            for condition in conditions:
                                if condition["operator"] == ">":
                                    combined_mask |= (col_data > condition["value"])
                                elif condition["operator"] == "<":
                                    combined_mask |= (col_data < condition["value"])
                                elif condition["operator"] == ">=":
                                    combined_mask |= (col_data >= condition["value"])
                                elif condition["operator"] == "<=":
                                    combined_mask |= (col_data <= condition["value"])
                            
                            total_matching = combined_mask.sum()
                            st.success(f"**Combined (OR logic): {total_matching} records match any condition**")
                        except:
                            pass
                        
                        thresholds[col] = {"type": "multiple_conditions_or", "conditions": conditions}
                    
                    else:  # Range
                        num_divisions = st.number_input(
                            f"Number of range divisions for {col}",
                            min_value=2,
                            max_value=20,
                            value=5,
                            key=f"range_divisions_{col}"
                        )
                        
                        # Calculate ranges
                        min_val = col_data.min()
                        max_val = col_data.max()
                        range_size = (max_val - min_val) / num_divisions
                        
                        ranges = []
                        for i in range(num_divisions):
                            start = min_val + (i * range_size)
                            if i == num_divisions - 1:  # Last range gets any remainder
                                end = max_val
                            else:
                                end = min_val + ((i + 1) * range_size)
                            ranges.append((start, end))
                        
                        st.info(f"Generated {num_divisions} ranges:")
                        for i, (start, end) in enumerate(ranges):
                            st.write(f"Range {i+1}: {start:.2f} to {end:.2f}")
                        
                        thresholds[col] = {"type": "range", "ranges": ranges}
                
                else:
                    # Show statistics for categorical columns
                    unique_values = col_data.unique()
                    value_counts = col_data.value_counts()
                    mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "N/A"
                    
                    st.write("**Unique Values:**")
                    st.write(f"Total unique: {len(unique_values)}")
                    st.write("**Value Distribution:**")
                    st.dataframe(value_counts.head(10))
                    st.write(f"**Most Frequent Value:** {mode_value}")
                    
                    # Selection method for categorical data
                    selection_method = st.selectbox(
                        f"Value selection method for {col}",
                        ["Select Specific Values", "Select All Values", "Select Top N Values"],
                        key=f"selection_method_{col}"
                    )
                    
                    if selection_method == "Select Specific Values":
                        # For categorical data, let user select specific values
                        selected_values = st.multiselect(
                            f"Select values to include for {col}",
                            options=unique_values.tolist(),
                            default=[mode_value] if mode_value != "N/A" else [],
                            key=f"selected_values_{col}"
                        )
                        
                        if selected_values:
                            # Create multiple selection options for combinations
                            value_groups = []
                            for i, value in enumerate(selected_values):
                                value_groups.append([value])
                            
                            # Also add option for all selected values together
                            if len(selected_values) > 1:
                                value_groups.append(selected_values)
                            
                            thresholds[col] = {"type": "categorical", "value_groups": value_groups}
                        else:
                            thresholds[col] = {"type": "categorical", "value_groups": []}
                    
                    elif selection_method == "Select All Values":
                        # Use all unique values - ONLY as individual groups, no combining
                        value_groups = []
                        # Each value as individual group
                        for value in unique_values:
                            value_groups.append([value])
                        # Remove this line: value_groups.append(unique_values.tolist())
                        
                        thresholds[col] = {"type": "categorical", "value_groups": value_groups}
                        st.info(f"Analyzing all {len(unique_values)} values individually")
                    
                    else:  # Select Top N Values
                        top_n = st.number_input(
                            f"Number of top values to include for {col}",
                            min_value=1,
                            max_value=len(unique_values),
                            value=min(5, len(unique_values)),
                            key=f"top_n_{col}"
                        )
                        
                        top_values = value_counts.head(top_n).index.tolist()
                        st.info(f"Selected top {top_n} values: {top_values}")
                        
                        value_groups = []
                        # Each top value as individual group - no combining
                        for value in top_values:
                            value_groups.append([value])
                        # Remove this: if len(top_values) > 1: value_groups.append(top_values)
                        
                        thresholds[col] = {"type": "categorical", "value_groups": value_groups}

        # Step 4: Select ID column and result columns
        st.markdown('<h2 class="section-header">Output Configuration</h2>', unsafe_allow_html=True)
        
        # ID column selection
        id_column = st.selectbox("Select identifier column", columns)
        
        # Result columns selection
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Select all toggle
        select_all_results = st.checkbox("Select all numeric columns for analysis")
        
        if select_all_results:
            result_columns = numeric_columns
            st.info(f"Selected all {len(numeric_columns)} numeric columns for analysis")
        else:
            result_columns = st.multiselect(
                "Select columns for statistical calculations", 
                numeric_columns,
                default=[]
            )

        # Step 5: Run analysis
        st.markdown('<h2 class="section-header">Execute Analysis</h2>', unsafe_allow_html=True)
        st.markdown("---")

        # Add this after the result columns selection and before the analyze button
        st.subheader("Analysis Settings")
        min_matching_rows = st.number_input(
            "Minimum Matching Rows Threshold",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            help="Filter out combinations with fewer than this many matching rows to focus on statistically relevant results"
        )
        
        if st.button("Analyze Data Combinations", type="primary"):
            if not selected_columns:
                st.error("Please select at least one column for analysis")
            elif not result_columns:
                st.error("Please select at least one result column")
            else:
                with st.spinner("Analyzing data combinations..."):
                    try:
                        # Pass the min_matching_rows parameter to the analysis function
                        results = analyze_data_combinations(
                            df, 
                            selected_columns, 
                            thresholds, 
                            id_column, 
                            result_columns,
                            min_matching_rows=min_matching_rows
                        )
                        
                        st.markdown('<h3 style="color: #374151;">Result of Analysis</h3>', unsafe_allow_html=True)
                        st.dataframe(results)

                        # Step 6: Download results
                        if not results.empty:
                            if st.button("Download Results", type="secondary"):
                                export_results(results)
                                st.success("Results exported successfully")
                        else:
                            st.warning("No combinations produced results")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        else:
            st.info("Adjust settings and click 'Run Analysis' to view results")
