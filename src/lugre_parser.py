import re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_unique_timestamps(dir: Path) -> list[str]:
    """
    Get unique timestamps from LuGRE data directory.
    Args:
        dir (Path): Path to the LuGRE data directory.
    Returns:
        list[str]: Sorted list of unique timestamps.
    """
    pattern = re.compile(r"TLM_NAV_(\d{8}_\d{6}_[0-9A-Z]+_.?_OP\d+_\d+)")
    unique_timestamp = set()

    for file in dir.iterdir():
        match = pattern.search(file.name)
        if match:
            unique_timestamp.add(match.group(1))

    return sorted(unique_timestamp)

def get_const_band(signalId: int):
    """
    Map signal ID to constellation and frequency band.
    Args:
        signalId (int): Signal ID number.
    Returns:
        tuple: (constellation, frequency band)
    """
    # Mapping of signal IDs to constellations and frequency bands
    mapping = {
        0: ("G", "L1CA"),   # 0: GPS L1 C/A
        1: ("G", "L5"),     # 1: GPS L5
        2: ("E", "E1BC"),   # 2: Galileo E1BC
        3: ("E", "E5A"),    # 3: Galileo E5A
        4: ("E", "E5B"),    # 4: Galileo E5B
    }

    return mapping.get(signalId, ("Unknown", "Unknown"))

def import_txt_file(file):
    """
    Import LuGRE data from a text file into a dataframe.
    Args:
        file (str or Path): Path to the text file
    Returns:
        pd.DataFrame: DataFrame containing the imported LuGRE data.
    """
    pattern = re.compile(r'(\w+):\s+([^\s]+)')
    data = []

    with open(file, 'r') as f:
        for line in f:
            matches = pattern.findall(line)
            if matches:
                entry = {key: value for key, value in matches}
                data.append(entry)
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def lugre_parser(path: Path, set_id: str) -> dict[str, pd.DataFrame]:
    """
    Parse LuGRE data files from a specified directory and set ID.
    Args:
        path (Path): Path to the LuGRE data directory.
        set_id (str): Set ID for the LuGRE data files.
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing DataFrames for each LuGRE data type.
    """
    # create info dataframe from set_id
    info = set_id.split('_')
    info_df = pd.DataFrame([info], columns=['date', 'time', 'duration', 'phase', 'opnumber', 'index'])

    # construct file paths
    acq_file = path.joinpath(f"TLM_ACQ_{set_id}.txt")
    clk_file = path.joinpath(f"TLM_CLK_{set_id}.csv")
    eph_file = path.joinpath(f"TLM_EPH_{set_id}.csv")
    nav_file = path.joinpath(f"TLM_NAV_{set_id}.txt")
    raw_file = path.joinpath(f"TLM_RAW_{set_id}.txt")
    # check if files exist
    for file in [acq_file, clk_file, eph_file, nav_file, raw_file]:
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist.")

    acq_df = import_txt_file(acq_file)
    nav_df = import_txt_file(nav_file)
    eph_df = pd.read_csv(eph_file)
    clk_df = pd.read_csv(clk_file)
    raw_df = import_txt_file(raw_file)
    # construct dictionary of dataframes
    lugre_df = {"info": info_df, "acq": acq_df, "nav": nav_df, "eph": eph_df, "clk": clk_df, "raw": raw_df}
    return lugre_df

def get_unique_times(lugre_df: pd.DataFrame, minmax: bool = True) -> np.ndarray:
    """
    Get all unique timestamps for all subfiles from LuGRE DataFrame.
    Args:
        lugre_df (pd.DataFrame): Dictionary of DataFrames containing LuGRE data.
        minmax (bool): If True, also return earliest and latest timestamps with source info.
    Returns:
        np.ndarray: Array of unique timestamps. If minmax is True, also returns
                    tuples with earliest and latest timestamps and their sources.
    """
    # get earliest and latest time of dataset
    time_keys = ['rxTime', 'Receiver Time [s]']

    rxTime_start, rxTime_end = float('inf'), float('-inf')
    rxTimes = np.array([])  # also get all times in one array
    src_min, src_max = None, None
    for name, df in lugre_df.items():
        for key in time_keys:
            if key in df.columns:
                t_set_min = float(df[key].min())
                t_set_max = float(df[key].max())
                if t_set_min < rxTime_start:
                    rxTime_start = t_set_min
                    src_min = name
                if t_set_max > rxTime_end:
                    rxTime_end = t_set_max
                    src_max = name
                rxTimes = np.append(rxTimes, df[key])

    rxTimes = np.sort(np.unique(rxTimes))

    if minmax:
        return rxTimes, (rxTime_start, src_min), (rxTime_end, src_max)
    else:
        return rxTimes
    
def gps_seconds_to_gps_weeks(gps_seconds: float) -> tuple[int, float]:
    """
    Convert GPS seconds to GPS weeks and seconds of week.
    Args:
        gps_seconds (np.ndarray): Array of GPS seconds.
    Returns:
        tuple: (gps_weeks (int), seconds_of_week (float))
    """
    sec_per_week = 604800  # Number of seconds in a GPS week
    gps_weeks = int(gps_seconds // sec_per_week)
    seconds_of_week = gps_seconds % sec_per_week
    return gps_weeks, seconds_of_week

def utc_round(dt: datetime) -> datetime:
    # If microseconds are >= 500_000, round up by adding 1 second
    if dt.microsecond >= 500_000:
        dt += timedelta(seconds=1)
    # Remove microseconds
    return dt.replace(microsecond=0)