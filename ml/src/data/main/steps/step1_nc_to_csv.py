import os
import netCDF4 as nc
import pandas as pd
import logging
from tqdm import tqdm

def convert_nc_to_csv(nc_file, csv_file):
    """Convert NetCDF file to CSV"""
    try:
        ds = nc.Dataset(nc_file)
        data_dict = {}

        # Debug information for variables
        for var in ds.variables:
            data = ds.variables[var][:]
            if data.ndim == 1:
                data_dict[var] = data
            elif data.ndim == 2:
                for i in range(data.shape[1]):
                    data_dict[f"{var}_col{i}"] = data[:, i]
            else:
                logging.debug(f"Skipping variable '{var}' because it is not 1- or 2-dimensional.")

        df = pd.DataFrame(data_dict)
        df.to_csv(csv_file, index=False)
        logging.info(f"Converted {nc_file} to {csv_file}")
        return True
    except Exception as e:
        logging.error(f"Error converting {nc_file} to CSV: {str(e)}")
        return False

def convert_all_nc_to_csv(base_dir, start_year, end_year):
    """Convert all NetCDF files to CSV in the specified directory"""
    converted_count = 0
    all_nc_files = []
    
    # First, collect all NC files
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            continue
        for month in range(1, 13):
            month_str = f"{month:02d}"
            month_dir = os.path.join(year_dir, month_str)
            if not os.path.exists(month_dir):
                continue
            for root, dirs, files in os.walk(month_dir):
                for file in files:
                    if file.endswith(".nc"):
                        nc_file = os.path.join(root, file)
                        csv_file = os.path.splitext(nc_file)[0] + ".csv"
                        if not os.path.exists(csv_file):
                            all_nc_files.append((nc_file, csv_file))
    
    # Then process them with a progress bar
    for nc_file, csv_file in tqdm(all_nc_files, desc="Converting NetCDF to CSV"):
        if convert_nc_to_csv(nc_file, csv_file):
            converted_count += 1
    
    logging.info(f"Converted {converted_count} NetCDF files to CSV")
    return converted_count
