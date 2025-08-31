import pandas as pd

def load_and_process_data(uploaded_file):
    """Load and process the uploaded Excel file"""
    try:
        df = pd.read_excel(uploaded_file, header=0)
        print(f"After reading Excel: {len(df)} rows")
        
        empty_rows = df.isnull().all(axis=1).sum()
        print(f"Completely empty rows found: {empty_rows}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading Excel file: {str(e)}")