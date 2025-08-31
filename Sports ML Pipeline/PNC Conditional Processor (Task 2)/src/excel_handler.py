import pandas as pd
import streamlit as st
from io import BytesIO
import datetime

def export_results(results_df, filename=None):
    """
    Export results DataFrame to Excel and provide download link
    
    Args:
        results_df: DataFrame with analysis results
        filename: Optional custom filename
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.xlsx"
    
    # Create Excel file in memory
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Analysis Results']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with formatting
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            # Auto-adjust column width
            worksheet.set_column(col_num, col_num, max(len(value) + 2, 15))
        
        # Append summary rows: Count and Sum
        try:
            start_row = len(results_df) + 1  # header is row 0, data starts at row 1

            # Compute counts (non-null counts) and sums (numeric only)
            counts = results_df.count()
            sums = results_df.select_dtypes(include=["number"]).sum()
            
            # Add variance calculation for numeric columns
            variances = results_df.select_dtypes(include=["number"]).var()

            label_format = workbook.add_format({'bold': True, 'border': 1, 'fg_color': '#F2F2F2'})
            value_format = workbook.add_format({'border': 1})

            # Write 'Count' row
            worksheet.write(start_row, 0, 'Count', label_format)
            for col_idx, col_name in enumerate(results_df.columns):
                # write counts for every column
                val = counts.get(col_name, "")
                # counts should be written starting at same column position
                worksheet.write(start_row, col_idx, val, value_format)

            # Write 'Sum' row
            worksheet.write(start_row + 1, 0, 'Sum', label_format)
            for col_idx, col_name in enumerate(results_df.columns):
                # only write sums for numeric columns, else leave blank
                if col_name in sums.index:
                    worksheet.write(start_row + 1, col_idx, sums[col_name], value_format)
                else:
                    worksheet.write(start_row + 1, col_idx, "", value_format)
                    
            # Write 'Variance' row
            worksheet.write(start_row + 2, 0, 'Variance', label_format)
            for col_idx, col_name in enumerate(results_df.columns):
                # only write variance for numeric columns, else leave blank
                if col_name in variances.index:
                    worksheet.write(start_row + 2, col_idx, variances[col_name], value_format)
                else:
                    worksheet.write(start_row + 2, col_idx, "", value_format)
        except Exception:
            # if anything goes wrong with summary rows, skip silently
            pass
    
    # Reset buffer position
    output.seek(0)
    
    # Provide download button
    st.download_button(
        label="📥 Download Excel File",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results"
    )

def save_results_locally(results_df, filepath):
    """
    Save results to local file (for testing purposes)
    """
    try:
        results_df.to_excel(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False