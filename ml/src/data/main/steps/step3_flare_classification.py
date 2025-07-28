import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from calendar import monthrange

def classify_flare(flux):
    """Classify solar flares based on xrsb_flux"""
    if flux > 1e-4:
        return 4  # X-class
    elif flux > 1e-5:
        return 3  # M-class
    elif flux > 1e-6:
        return 2  # C-class
    elif flux > 0:
        return 1  # B-class or smaller
    else:
        return 0  # No data or invalid

def check_missing_data(sub_df, required_minutes=1440):  # 24 hours * 60 minutes
    """Check for missing data"""
    if sub_df["time"].isnull().any():
        return True

    start_time = sub_df["time"].min()
    if pd.isnull(start_time):
        return True
    
    expected_times = pd.date_range(start=start_time, periods=required_minutes, freq="1min")
    actual_times = pd.to_datetime(sub_df["time"])
    missing_minutes = expected_times.difference(actual_times)
    return len(missing_minutes) > 8 * 60  # Allow up to 8 hours of missing data

def process_day_for_flare_class(year, month, day, base_dir):
    """Process a single day's data for flare classification"""
    year_str = str(year)
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    file_path = os.path.join(base_dir, year_str, month_str, f"corrected_{year_str}_{month_str}_{day_str}.csv")

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
    df["flare_class"] = 0  # Default to 0 (no data)

    # Filter to only keep hourly data (minute == 0)
    hourly_df = df[df["time"].dt.minute == 0].copy().reset_index(drop=True)

    for i in range(len(hourly_df)):
        start_time = hourly_df.loc[i, "time"]
        end_time = start_time + pd.Timedelta(hours=24)

        # Collect the next 24 hours of data
        next_24_hours = df[(df["time"] >= start_time) & (df["time"] < end_time)]

        # Check if we need data from the next day
        if end_time.day != start_time.day:
            next_day = start_time + pd.Timedelta(days=1)
            next_day_file = os.path.join(
                base_dir, 
                f"{next_day.year}/{next_day.month:02d}/corrected_{next_day.year}_{next_day.month:02d}_{next_day.day:02d}.csv"
            )
            
            if os.path.exists(next_day_file):
                next_day_df = pd.read_csv(next_day_file)
                next_day_df["time"] = pd.to_datetime(next_day_df["time"], errors="coerce").dt.tz_localize(None)
                next_day_data = next_day_df[next_day_df["time"] < end_time]
                next_24_hours = pd.concat([next_24_hours, next_day_data])

        if check_missing_data(next_24_hours, required_minutes=1440):
            hourly_df.loc[i, "flare_class"] = 0  # Missing data
        else:
            max_flux = next_24_hours["xrsb_flux"].max()
            if pd.notna(max_flux):  # Check if max_flux is not NaN
                hourly_df.loc[i, "flare_class"] = classify_flare(max_flux)
                if classify_flare(max_flux) == 4:  # X-class
                    logging.info(f"X-class flare detected on {start_time}, max flux: {max_flux}")

    new_file_path = os.path.join(base_dir, year_str, month_str, f"complete_{year_str}_{month_str}_{day_str}.csv")
    hourly_df.to_csv(new_file_path, index=False)
    logging.info(f"Processed file: {file_path}")

    return new_file_path

def process_all_days_for_flare_class(base_dir, start_year, end_year):
    """Process all days for flare classification"""
    processed_count = 0
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            days_in_month = monthrange(year, month)[1]
            
            # Create a list of days to process
            days_to_process = []
            for day in range(1, days_in_month + 1):
                corrected_file = os.path.join(
                    base_dir, 
                    f"{year}/{month:02d}/corrected_{year}_{month:02d}_{day:02d}.csv"
                )
                complete_file = os.path.join(
                    base_dir, 
                    f"{year}/{month:02d}/complete_{year}_{month:02d}_{day:02d}.csv"
                )
                
                if os.path.exists(corrected_file) and not os.path.exists(complete_file):
                    days_to_process.append((year, month, day))
            
            # Process days with progress bar
            for year, month, day in tqdm(days_to_process, 
                                        desc=f"Processing flare classification for {year}-{month:02d}"):
                try:
                    process_day_for_flare_class(year, month, day, base_dir)
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Error processing day {year}-{month:02d}-{day:02d}: {e}")
    
    logging.info(f"Processed flare classification for {processed_count} days")
    return processed_count
