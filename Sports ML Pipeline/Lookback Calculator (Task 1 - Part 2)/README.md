# Football Match Rolling Averages Calculator

A Streamlit application that processes football match data to compute rolling averages based on team performance, favorite status, and match results.

## Features

- Upload CSV files with match data
- Compute rolling averages for numeric statistics
- Support for multiple conditions:
  - Team Result (Win/Lose/Draw)
  - Favorite status (Favorite/Underdog)
  - Rolling windows (Last 5, Last 10, All previous matches)
- Download processed data as CSV
- Interactive data preview

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## CSV Format

Your CSV file should contain these columns:
- **MatchID**: Unique identifier for each match
- **Date**: Match date (YYYY-MM-DD format preferred)
- **Team**: Team name
- **Team_Result**: Win/Lose/Draw
- **IsFavorite**: True/False
- **Numeric columns**: HS, HF, AS, AF, etc. (any numeric stats you want rolling averages for)

Each match should have 2 rows (one for each team).

## Generated Columns

The app generates rolling average columns in the format:
- `{result}_avg_{favorite_status}_{stat_column}_{window}`

Examples:
- `win_avg_fav_HS_L5`: Average HS for last 5 wins when team was favorite
- `lose_avg_underdog_AF_L10`: Average AF for last 10 losses when team was underdog
- `draw_avg_fav_HF_All`: Average HF for all previous draws when team was favorite

## Notes

- Rolling averages are computed based on **previous matches only** (no future leakage)
- If fewer matches exist than the window size, the value is set to 0
- Processing is optimized for large datasets using efficient pandas operations
