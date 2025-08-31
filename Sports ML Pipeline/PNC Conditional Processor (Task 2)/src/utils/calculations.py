def calculate_mean(series):
    return series.mean()

def calculate_median(series):
    return series.median()

def calculate_mode(series):
    return series.mode().iloc[0] if not series.mode().empty else None

def calculate_std(series):
    return series.std()

def calculate_variance(series):
    return series.var()

def calculate_percentile(series, percentile):
    return series.quantile(percentile / 100)

def calculate_summary_statistics(series):
    return {
        "mean": calculate_mean(series),
        "median": calculate_median(series),
        "mode": calculate_mode(series),
        "std_dev": calculate_std(series),
        "variance": calculate_variance(series),
    }