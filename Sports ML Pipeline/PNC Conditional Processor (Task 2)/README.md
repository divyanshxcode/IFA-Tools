# Data Analysis Tool

This project is a data analysis tool built using Streamlit that allows users to upload Excel datasets, select columns for analysis, define thresholds, and download the results. 

## Features

- Upload any Excel dataset.
- Select specific columns for analysis.
- Define thresholds for numeric and categorical data.
- Analyze data based on user-defined conditions.
- Download the results as an Excel file.

## Project Structure

```
data-analysis-tool
├── src
│   ├── __init__.py
│   ├── app.py                  # Main entry point for the Streamlit application
│   ├── data_processor.py       # Functions for loading and processing the dataset
│   ├── analysis_engine.py       # Core logic for analyzing the data
│   ├── excel_handler.py        # Manages exporting results to Excel
│   ├── filter_manager.py       # Functions for managing and applying filters
│   └── utils
│       ├── __init__.py
│       ├── calculations.py      # Utility functions for calculations
│       └── validators.py        # Functions for validating user inputs
├── requirements.txt            # Project dependencies
├── config.py                   # Configuration settings
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd data-analysis-tool
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open the application in your web browser (usually at `http://localhost:8501`).

3. Upload your Excel data file.

4. Select the columns you want to analyze.

5. Define thresholds for each selected column.

6. Click "Analyze" to see the results.

7. Download the results if needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you would like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for details.