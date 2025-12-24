from datetime import datetime
import pandas as pd

is_in_range = lambda x, bounds, if_reverse=False: (
        bounds[1] <= x <= bounds[0] if if_reverse else bounds[0] <= x <= bounds[1]
    )

def decimal_arange(start, end, step, decimal=1):
    """Generates a list of numbers with exactly one decimal precision, including the end"""
    return [round(start + i * step, decimal) for i in range(int((end - start) / step) + 1)]


def batch_strtime2sec(time_series: pd.Series, start_time: datetime, time_format: str) -> pd.Series:
    """
    Convert a Series of datetime strings (in HHMMSSFFF format) into seconds relative to the start time.

    Args:
        time_series (pd.Series): A Pandas Series of datetime strings in HHMMSSFFF format.
        start_time (datetime): The reference start time as a datetime object.

    Returns:
        pd.Series: A Pandas Series of seconds relative to the start time.
    """

    time_series = time_series.astype(str)

    base_time = datetime.today()
    base_time_format = "%Y%m%d"
    # Parse the time strings into datetime objects relative to the base date
    time_objs = pd.to_datetime(base_time.strftime(base_time_format) + time_series, format=base_time_format + time_format)
    start_time = datetime.combine(base_time, start_time)

    # Calculate seconds relative to the start time
    delta_seconds = (time_objs - start_time).dt.total_seconds()

    return delta_seconds

def strtime2sec(time: str, start_time: datetime, time_format: str, base_time: datetime) -> float:
    """
    Convert the given datetime string (in HHMMSSFFF format) into seconds,
    relative to the start time.

    Args:
        time (str): The datetime string in HHMMSSFFF format (e.g., '070216100').
        start_time (datetime): The minimum datetime string (start time in HHMMSSFFF format).

    Returns:
        int: The time in seconds relative to the start time.
    """
    time = str(time)

    # Parse the input time into a time object
    time_part = datetime.strptime(time, time_format).time()

    # Combine the base date with the parsed time
    time_obj = datetime.combine(base_time.date(), time_part)

    # Calculate the difference in seconds relative to the start time
    delta_seconds = (time_obj - start_time).total_seconds()

    return delta_seconds


def strtime2datetime(time: str, time_format: str) -> datetime:
    """
    Convert a time string in HHMMSSFFF format into a datetime object.

    Args:
        time (str): The time string in HHMMSSFFF format (e.g., '070216100').

    Returns:
        datetime: A datetime object representing the input time.
    """
    time = str(time)
    # Define the format for parsing the datetime string
    
    # Combine the base date with the parsed time
    return datetime.strptime(time, time_format).time()  # Extract time part only


min2sec = lambda x: x * 60
sec2min = lambda x: x / 60