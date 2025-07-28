import os
import re
import pandas as pd
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

def extract_date_from_filename(filename):
    """Extract date information from filename"""
    match = re.search(r"d(\d{8})", filename)
    if match:
        date_str = match.group(1)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return year, month, day
    else:
        raise ValueError(f"Invalid date format in filename: {filename}")

def get_next_date(year, month, day):
    """Get the next day's date"""
    date = datetime(year, month, day)
    next_date = date + timedelta(days=1)
    return next_date

def load_and_convert_csv(csv_file):
    """Load and convert CSV file, transforming time column"""
    df = pd.read_csv(csv_file)
    if "time" in df.columns:
        base_date = datetime(2000, 1, 1, 12, 0, 0)  # Set base date
        df["time"] = pd.to_datetime(df["time"], unit="s", origin=base_date)
        df["time"] = df["time"].dt.tz_localize("UTC")  # Set timezone to UTC
        df["time"] = df["time"].dt.tz_convert("UTC")  # Convert timezone to UTC
        df["time"] = df["time"].dt.tz_localize(None)  # Remove timezone information
        return df
    else:
        raise ValueError(f"'time' column not found in {csv_file}")

def save_complemented_data(year, month, day, base_dir, combined_df):
    """Save complemented data"""
    output_dir = os.path.join(base_dir, str(year), f"{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"corrected_{year}_{month:02d}_{day:02d}.csv"
    )

    expected_minutes = set(range(1440))  # Expected minutes in 24 hours
    actual_minutes = set(
        (combined_df["time"] - datetime(year, month, day)).dt.total_seconds() // 60
    )
    missing_minutes = expected_minutes - actual_minutes

    for minute in missing_minutes:
        missing_time = datetime(year, month, day) + timedelta(minutes=minute)
        missing_row = pd.Series(
            [None] * len(combined_df.columns), index=combined_df.columns
        )
        missing_row["time"] = missing_time
        combined_df = pd.concat(
            [combined_df, pd.DataFrame([missing_row])], ignore_index=True
        )

    combined_df["time"] = pd.to_datetime(combined_df["time"]).dt.tz_localize(
        None
    )  # Remove timezone
    combined_df = combined_df.sort_values(by="time").reset_index(drop=True)

    # Only save data for the current date
    combined_df = combined_df[
        (combined_df["time"] >= datetime(year, month, day))
        & (combined_df["time"] < datetime(year, month, day) + timedelta(days=1))
    ]

    combined_df.to_csv(output_file, index=False)
    logging.info(f"Saved {output_file}")
    return output_file

def process_csv_files(base_dir):
    """Process CSV files to create complete day data"""
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv") and ("v1-0-0" in file or "v2-2-0" in file):
                all_files.append(os.path.join(root, file))

    all_files.sort()
    processed_count = 0
    
    for csv_file in tqdm(all_files, desc="Processing CSV files"):
        try:
            # Extract date information from filename
            year, month, day = extract_date_from_filename(os.path.basename(csv_file))

            # Load current day's data
            df = load_and_convert_csv(csv_file)

            # Load next day's data
            next_date = get_next_date(year, month, day)
            next_csv_file = None

            # Try to find next day's file with either version
            for version in ["v1-0-0", "v2-2-0"]:
                potential_file = os.path.join(
                    base_dir,
                    str(next_date.year),
                    f"{next_date.month:02d}",
                    f"sci_xrsf-l2-avg1m_g15_d{next_date.year}{next_date.month:02d}{next_date.day:02d}_{version}.csv",
                )
                if os.path.exists(potential_file):
                    next_csv_file = potential_file
                    break

            combined_df = df.copy()

            if next_csv_file and os.path.exists(next_csv_file):
                next_df = load_and_convert_csv(next_csv_file)
                combined_df = pd.concat([combined_df, next_df], ignore_index=True)

            save_complemented_data(year, month, day, base_dir, combined_df)
            processed_count += 1

        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    logging.info(f"Processed {processed_count} CSV files")
    return processed_count
