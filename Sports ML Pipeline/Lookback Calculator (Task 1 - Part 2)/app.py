import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time

def compute_rolling_averages(df, selected_numeric_cols, selected_favorite_conditions, selected_result_conditions):
    """
    Compute rolling averages for each team based on previous matches only.
    """
    # Sort by Team and Date to ensure chronological order
    df = df.sort_values(['Team', 'Date', 'MatchID']).reset_index(drop=True)
    
    # Pre-compute all column names that will be added
    windows = {'L5': 5, 'L10': 10, 'All': None}
    
    new_columns = []
    
    # Generate column names based on user selections
    if selected_result_conditions and selected_favorite_conditions:
        # Result-specific and favorite-specific averages
        for result_key in selected_result_conditions.keys():
            for fav_key in selected_favorite_conditions.keys():
                for window_key in windows.keys():
                    for col in selected_numeric_cols:
                        new_columns.append(f"{result_key}_avg_{fav_key}_{col}_{window_key}")
    
    if selected_favorite_conditions:
        # Overall averages by favorite status (without result filter)
        for fav_key in selected_favorite_conditions.keys():
            for window_key in windows.keys():
                for col in selected_numeric_cols:
                    new_columns.append(f"avg_{fav_key}_{col}_{window_key}")
    
    if not selected_favorite_conditions:
        # Overall averages (no favorite or result filters)
        for window_key in windows.keys():
            for col in selected_numeric_cols:
                new_columns.append(f"avg_{col}_{window_key}")
    
    # Initialize result dataframe with all new columns at once
    result_df = df.copy()
    for col in new_columns:
        result_df[col] = 0.0
    
    # Process each team separately
    teams = df['Team'].unique()
    total_rows = len(df)
    processed_rows = 0
    
    # Create progress bars
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for team_idx, team in enumerate(teams):
        team_data = df[df['Team'] == team].copy()
        
        # For each row in the team's data
        for idx in range(len(team_data)):
            current_row_idx = team_data.index[idx]
            
            # Get all previous matches for this team (before current date/match)
            prev_matches = team_data.iloc[:idx]  # All rows before current in sorted order
            
            if len(prev_matches) > 0:
                # Compute result-specific and favorite-specific averages
                if selected_result_conditions and selected_favorite_conditions:
                    for result_key, result_value in selected_result_conditions.items():
                        for fav_key, fav_value in selected_favorite_conditions.items():
                            # Filter previous matches by result and favorite status
                            filtered_matches = prev_matches[
                                (prev_matches['Team_Result'] == result_value) & 
                                (prev_matches['IsFavorite'] == fav_value)
                            ]
                            
                            # For each window size
                            for window_key, window_size in windows.items():
                                # Apply window
                                if window_size is None:  # 'All' case
                                    windowed_matches = filtered_matches
                                else:
                                    windowed_matches = filtered_matches.tail(window_size)
                                
                                # Compute averages for each selected numeric column
                                for col in selected_numeric_cols:
                                    if col in windowed_matches.columns:
                                        col_name = f"{result_key}_avg_{fav_key}_{col}_{window_key}"
                                        
                                        if len(windowed_matches) >= (window_size if window_size else 1):
                                            avg_value = windowed_matches[col].mean()
                                            result_df.at[current_row_idx, col_name] = avg_value
                
                # Compute overall averages by favorite status (without result filter)
                if selected_favorite_conditions:
                    for fav_key, fav_value in selected_favorite_conditions.items():
                        filtered_matches = prev_matches[prev_matches['IsFavorite'] == fav_value]
                        
                        for window_key, window_size in windows.items():
                            if window_size is None:
                                windowed_matches = filtered_matches
                            else:
                                windowed_matches = filtered_matches.tail(window_size)
                            
                            for col in selected_numeric_cols:
                                if col in windowed_matches.columns:
                                    col_name = f"avg_{fav_key}_{col}_{window_key}"
                                    
                                    if len(windowed_matches) >= (window_size if window_size else 1):
                                        avg_value = windowed_matches[col].mean()
                                        result_df.at[current_row_idx, col_name] = avg_value
                
                # Compute overall averages (no filters)
                if not selected_favorite_conditions:
                    for window_key, window_size in windows.items():
                        if window_size is None:
                            windowed_matches = prev_matches
                        else:
                            windowed_matches = prev_matches.tail(window_size)
                        
                        for col in selected_numeric_cols:
                            if col in windowed_matches.columns:
                                col_name = f"avg_{col}_{window_key}"
                                
                                if len(windowed_matches) >= (window_size if window_size else 1):
                                    avg_value = windowed_matches[col].mean()
                                    result_df.at[current_row_idx, col_name] = avg_value
            
            # Update progress
            processed_rows += 1
            progress_percentage = processed_rows / total_rows
            progress_bar.progress(progress_percentage)
            status_text.text(f"Processing: {processed_rows}/{total_rows} rows ({progress_percentage:.1%}) | Current team: {team}")
    
    # Clear status text
    status_text.text("✅ Processing completed!")
    
    return result_df

def main():
    st.title("📊 Rolling Averages Calculator")
    st.markdown("Upload your CSV file to compute rolling averages based on team performance, conditions, and match results.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with data containing columns like Team, Date, MatchID, Team_Result, IsFavorite, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show preview of uploaded data
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            # Validate required columns
            required_cols = ['Team', 'Date', 'MatchID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Required columns: Team, Date, MatchID")
                return
            
            # Convert Date to datetime if it's not already
            if df['Date'].dtype == 'object':
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Show data info
            st.subheader("📈 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Teams", df['Team'].nunique())
            with col3:
                st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            
            # Configuration Section
            st.subheader("⚙️ Configuration Options")
            
            # 1. Numeric Columns Selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['MatchID', 'Date']
            available_numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            st.write("**1. Select Numeric Columns for Rolling Averages:**")
            selected_numeric_cols = st.multiselect(
                "Choose columns to calculate rolling averages for:",
                options=available_numeric_cols,
                default=available_numeric_cols,  # Select all by default
                help="Select the numeric columns you want to calculate rolling averages for"
            )
            
            # 2. Favorite/Underdog Conditions (optional)
            st.write("**2. Filter by Favorite Status (optional):**")
            has_favorite_col = 'IsFavorite' in df.columns
            
            if has_favorite_col:
                favorite_options = st.multiselect(
                    "Select conditions to calculate averages for:",
                    options=['Favorite', 'Underdog', 'Both', 'None'],
                    default=['Both'],
                    help="Choose whether to calculate averages for favorites, underdogs, both, or ignore this filter"
                )
                
                # Convert to dictionary format
                selected_favorite_conditions = {}
                if 'Favorite' in favorite_options or 'Both' in favorite_options:
                    selected_favorite_conditions['fav'] = True
                if 'Underdog' in favorite_options or 'Both' in favorite_options:
                    selected_favorite_conditions['underdog'] = False
                if 'None' in favorite_options:
                    selected_favorite_conditions = {}
            else:
                st.info("No 'IsFavorite' column found. Skipping favorite/underdog filtering.")
                selected_favorite_conditions = {}
            
            # 3. Result Conditions (optional)
            st.write("**3. Filter by Results (optional):**")
            has_result_col = 'Team_Result' in df.columns
            
            if has_result_col:
                unique_results = df['Team_Result'].unique().tolist()
                
                result_options = st.multiselect(
                    "Select result types to calculate averages for:",
                    options=unique_results + ['All', 'None'],
                    default=['All'],
                    help="Choose which result types to calculate averages for, or select 'All' for all results, 'None' to ignore this filter"
                )
                
                # Convert to dictionary format
                selected_result_conditions = {}
                if 'All' in result_options:
                    for result in unique_results:
                        key = result.lower().replace(' ', '_')
                        selected_result_conditions[key] = result
                elif 'None' not in result_options:
                    for result in result_options:
                        if result not in ['All', 'None']:
                            key = result.lower().replace(' ', '_')
                            selected_result_conditions[key] = result
            else:
                st.info("No 'Team_Result' column found. Skipping result filtering.")
                selected_result_conditions = {}
            
            # Show configuration summary
            with st.expander("📋 Configuration Summary"):
                st.write(f"**Selected Numeric Columns:** {len(selected_numeric_cols)}")
                if selected_numeric_cols:
                    st.write(", ".join(selected_numeric_cols[:5]) + ("..." if len(selected_numeric_cols) > 5 else ""))
                
                st.write(f"**Favorite Conditions:** {list(selected_favorite_conditions.keys()) if selected_favorite_conditions else 'None'}")
                st.write(f"**Result Conditions:** {list(selected_result_conditions.keys()) if selected_result_conditions else 'None'}")
                
                # Estimate number of new columns
                windows = 3  # L5, L10, All
                estimated_cols = len(selected_numeric_cols) * windows
                
                if selected_result_conditions and selected_favorite_conditions:
                    estimated_cols *= len(selected_result_conditions) * len(selected_favorite_conditions)
                    estimated_cols += len(selected_numeric_cols) * windows * len(selected_favorite_conditions)
                elif selected_favorite_conditions:
                    estimated_cols *= len(selected_favorite_conditions)
                
                st.write(f"**Estimated New Columns:** ~{estimated_cols}")
            
            # Process button
            if selected_numeric_cols:  # Only show button if at least one numeric column is selected
                if st.button("🚀 Process Data & Compute Rolling Averages", type="primary"):
                    with st.spinner("Computing rolling averages... This may take a few minutes for large datasets."):
                        start_time = time.time()
                        
                        # Compute rolling averages
                        processed_df = compute_rolling_averages(
                            df, 
                            selected_numeric_cols, 
                            selected_favorite_conditions, 
                            selected_result_conditions
                        )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                    
                    st.success(f"✅ Processing completed in {processing_time:.2f} seconds!")
                    
                    # Show results
                    st.subheader("📋 Processed Data Preview")
                    st.dataframe(processed_df.head(10))
                    
                    st.subheader("📊 New Columns Added")
                    new_cols = [col for col in processed_df.columns if col not in df.columns]
                    st.info(f"Added {len(new_cols)} new rolling average columns")
                    
                    # Show some example new columns
                    if new_cols:
                        st.write("Sample new columns:")
                        for i, col in enumerate(new_cols[:10]):  # Show first 10
                            st.write(f"• {col}")
                        if len(new_cols) > 10:
                            st.write(f"... and {len(new_cols) - 10} more columns")
                    
                    # Download button
                    st.subheader("💾 Download Processed Data")
                    
                    # Convert to CSV for download
                    csv_buffer = BytesIO()
                    processed_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Processed CSV",
                        data=csv_data,
                        file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the dataset with all computed rolling averages"
                    )
                    
                    # Show summary statistics
                    st.subheader("📈 Summary Statistics")
                    st.write("Sample rolling averages for first team:")
                    first_team = processed_df['Team'].iloc[0]
                    team_sample = processed_df[processed_df['Team'] == first_team].head(5)
                    
                    # Show only the new columns for the sample
                    sample_new_cols = [col for col in team_sample.columns if col in new_cols]
                    if sample_new_cols:
                        basic_cols = ['Team', 'Date']
                        if 'Team_Result' in team_sample.columns:
                            basic_cols.append('Team_Result')
                        if 'IsFavorite' in team_sample.columns:
                            basic_cols.append('IsFavorite')
                        
                        st.dataframe(team_sample[basic_cols + sample_new_cols[:8]])
            else:
                st.warning("⚠️ Please select at least one numeric column to calculate rolling averages.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and required columns.")
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        # Show example format
        st.subheader("📋 Expected CSV Format")
        st.markdown("""
        Your CSV should contain these columns:
        - **Team**: Team/Entity identifier
        - **Date**: Date of the record (YYYY-MM-DD format preferred)
        - **MatchID**: Unique identifier for each record
        - **Team_Result**: Result category (optional - e.g., Win/Lose/Draw)
        - **IsFavorite**: Boolean condition (optional - True/False)
        - **Numeric columns**: Any numeric data you want rolling averages for
        
        The tool will calculate rolling averages (Last 5, Last 10, All-time) based on your selected conditions.
        """)

if __name__ == "__main__":
    main()
