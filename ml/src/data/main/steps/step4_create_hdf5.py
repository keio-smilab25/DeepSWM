import os
import numpy as np
import pandas as pd
import h5py
import cv2
from skimage.io import imread
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import matplotlib.pyplot as plt

# AIA and HMI processing parameters
AIA_VERTICAL_CROP = 20
AIA_HORIZONTAL_CROP = 15
HMI_TEXT_HEIGHT = 10
HMI_TEXT_WIDTH = 70

def read_fits(file_path):
    """Read FITS file and return data"""
    with fits.open(file_path) as hdul:
        data = hdul[1].data
    return data

def read_jpeg(file_path):
    """Read JPEG file and return data"""
    return imread(file_path, as_gray=True)

def process_aia_data(data, target_size=256, use_compression=True):
    """Process AIA data"""
    if target_size == 1024:
        # Keep original 1024x1024 size, no cropping
        pass
    else:
        # Crop the image to remove text for other sizes
        h, w = data.shape
        data = data[AIA_VERTICAL_CROP:h-AIA_VERTICAL_CROP, AIA_HORIZONTAL_CROP:w-AIA_HORIZONTAL_CROP]
        # Resize to target size
        data = cv2.resize(data, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32
    return data.astype(np.float32)

def process_hmi_data(data, target_size=256, use_compression=True):
    """Process HMI data"""
    if target_size == 1024:
        # Keep original 1024x1024 size, but remove bottom-left text
        # Text area is typically in bottom-left corner - increase width to fully cover text
        text_height = 60
        text_width = 350
        
        # Replace bottom-left text area with bottom-right pixels
        h, w = data.shape
        bottom_left_region = data[h-text_height:h, 0:text_width]
        bottom_right_region = data[h-text_height:h, w-text_width:w]
        
        # Copy bottom-right to bottom-left to remove text
        data[h-text_height:h, 0:text_width] = bottom_right_region
    else:
        # Crop the image to remove text for other sizes
        h, w = data.shape
        data = data[HMI_TEXT_HEIGHT:, HMI_TEXT_WIDTH:]
        # Resize to target size
        data = cv2.resize(data, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32
    return data.astype(np.float32)

def is_missing_data(data, threshold=0.01):
    """Check if data is missing (all zeros or very low values)"""
    return np.mean(data) < threshold

def find_past_similar_file(timestamp, processed_files):
    """Find past similar file for missing data"""
    if not processed_files:
        return None
    
    # Sort by time difference
    sorted_files = sorted(processed_files.items(), key=lambda x: abs((x[0] - timestamp).total_seconds()))
    return processed_files[sorted_files[0][0]]

def save_sample_images(X, y_label, timestamp, output_dir):
    """Save sample images"""
    sample_dir = os.path.join(
        output_dir, "visualization", timestamp.strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(sample_dir, exist_ok=True)

    # Define AIA wavelengths and corresponding colormaps
    aia_wavelengths_vis = ["94", "131", "171", "193", "211", "304", "335", "1600", "4500"]
    # Use standard matplotlib colormaps
    aia_colormaps = ["plasma", "viridis", "inferno", "magma", "cividis", "hot", "copper", "gray", "coolwarm"]

    # Save images for each channel
    for i in range(10):
        plt.figure(figsize=(8, 8))

        if i < 9:
            # Visualize AIA channels
            wavelength = aia_wavelengths_vis[i]
            title = f"AIA {wavelength} Å"
            cmap = aia_colormaps[i]

            # Apply normalization
            plt.imshow(X[i], cmap=cmap, vmin=np.percentile(X[i], 5), vmax=np.percentile(X[i], 95))

        else:
            # Visualize HMI
            title = "HMI Grayscale"
            plt.imshow(X[i], cmap="gray", vmin=np.percentile(X[i], 5), vmax=np.percentile(X[i], 95))

        plt.title(title)
        plt.colorbar()
        plt.axis("off")

        # Save to sample folder
        save_path = os.path.join(
            sample_dir, f'channel_{i:02d}_{title.replace(" ", "_")}.png'
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()

    # Create README in sample folder with class label information
    flare_class_names = {
        0: "No flare / Missing data",
        1: "B-class flare", 
        2: "C-class flare",
        3: "M-class flare",
        4: "X-class flare"
    }
    
    with open(os.path.join(sample_dir, "README.txt"), "w") as f:
        f.write(f"Sample: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Class Label: {y_label} ({flare_class_names.get(y_label, 'Unknown')})\n\n")
        f.write("Channel Information:\n")
        for i in range(9):
            f.write(f"Channel {i:02d}: AIA {aia_wavelengths_vis[i]} Å\n")
        f.write("Channel 09: HMI Magnetogram\n")

def process_hour(
    year, month, day, hour, 
    aia_base_dir, hmi_base_dir, xrs_base_dir, 
    aia_wavelengths, processed_files, 
    visualize=False, vis_dir=None, target_size=256, use_compression=True
):
    """Process one hour of data"""
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)

    # Process XRS data - updated to use 24-hour forward window
    xrs_file_path = os.path.join(
        xrs_base_dir, f"{year}/{month:02d}/complete_{year}_{month:02d}_{day:02d}.csv"
    )
    
    try:
        # Load XRS data
        xrs_df = pd.read_csv(xrs_file_path)
        xrs_df["time"] = pd.to_datetime(xrs_df["time"])
        
        # Find the current timestamp in the data
        current_time = timestamp
        current_row = xrs_df[xrs_df["time"] == current_time]
        
        if current_row.empty:
            y_label = 0  # Missing data
        else:
            y_label = current_row["flare_class"].values[0]
            
    except Exception as e:
        logging.error(f"Error processing XRS data for {timestamp}: {e}")
        y_label = 0

    # Calculate expected image size after processing
    if target_size == 1024:
        # Both AIA and HMI will be 1024x1024 (AIA: no crop, HMI: no crop + text removal)
        expected_aia_size = (1024, 1024)
        expected_hmi_size = (1024, 1024)
    else:
        expected_aia_size = (target_size, target_size)
        expected_hmi_size = (target_size, target_size)

    # Process AIA data
    X_data = []
    for wavelength in aia_wavelengths:
        aia_file = os.path.join(
            aia_base_dir, f"{year}/{month:02d}/{day:02d}/{wavelength}/{hour:02d}00.fits"
        )
        try:
            aia_data = read_fits(aia_file)
            aia_data = process_aia_data(aia_data, target_size=target_size, use_compression=use_compression)
            X_data.append(aia_data)
        except Exception as e:
            logging.error(f"Error processing AIA data for {wavelength}: {e}")
            X_data.append(np.zeros(expected_aia_size, dtype=np.float32))

    # Process HMI data
    hmi_file = os.path.join(
        hmi_base_dir, f"{year}/{month:02d}/{day:02d}/{hour:02d}00.jpg"
    )
    try:
        hmi_data = read_jpeg(hmi_file)
        hmi_processed = process_hmi_data(hmi_data, target_size=target_size, use_compression=use_compression)

        # If HMI data is missing, supplement with past data
        if is_missing_data(hmi_processed):
            past_data = find_past_similar_file(timestamp, processed_files)
            if past_data is not None:
                hmi_processed = past_data[-1]  # Last channel is HMI
    except Exception as e:
        logging.error(f"Error processing HMI data: {e}")
        hmi_processed = np.zeros(expected_hmi_size, dtype=np.float32)

    # Combine all channels
    X_combined = np.stack(X_data + [hmi_processed], axis=0)

    # In visualization mode
    if visualize and vis_dir is not None:
        save_sample_images(X_combined, y_label, timestamp, vis_dir)

    return X_combined, y_label, str(timestamp)

def process_date(date, aia_base_dir, hmi_base_dir, xrs_base_dir, output_dir, aia_wavelengths, mode="create", vis_dir=None, target_size=256, use_compression=True):
    """Function to process one hour of data"""
    try:
        X, y, _ = process_hour(
            date.year,
            date.month,
            date.day,
            date.hour,
            aia_base_dir, 
            hmi_base_dir, 
            xrs_base_dir,
            aia_wavelengths,
            {},  # processed_files not used in parallel processing
            visualize=(mode == "visualize"),
            vis_dir=vis_dir,
            target_size=target_size,
            use_compression=use_compression,
        )

        # File name format: YYYYMMDD_HHMMSS.h5
        timestamp_str = date.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{timestamp_str}.h5")

        with h5py.File(output_file, "w") as f:
            if use_compression:
                f.create_dataset("X", data=X, compression="gzip", compression_opts=9)
            else:
                f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
            f.create_dataset("timestamp", data=timestamp_str)

        return True
    except Exception as e:
        logging.error(f"Error processing {date}: {e}")
        return False

def create_hourly_hdf5_datasets(
    aia_base_dir, hmi_base_dir, xrs_base_dir, output_dir,
    start_date, end_date, aia_wavelengths,
    mode="create", vis_dir=None, num_workers=os.cpu_count(),
    target_size=256, use_compression=True
):
    """Create datasets (parallel processing version)"""
    os.makedirs(output_dir, exist_ok=True)

    # Create list of dates to process
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if mode == "visualize":
            # In visualize mode, 50 samples every 1000 hours
            date_list.append(current_date)
            current_date += pd.Timedelta(hours=1000)
            if len(date_list) >= 50:
                break
        else:
            # In create mode, every hour
            date_list.append(current_date)
            current_date += pd.Timedelta(hours=1)

    # Setup progress bar
    total = len(date_list)
    processed = 0
    pbar = tqdm(total=total, desc="Processing data")

    # Execute parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit jobs
        futures = [
            executor.submit(
                process_date, 
                date, 
                aia_base_dir, 
                hmi_base_dir, 
                xrs_base_dir, 
                output_dir, 
                aia_wavelengths,
                mode, 
                vis_dir,
                target_size,
                use_compression
            ) for date in date_list
        ]

        # Collect results
        for future in as_completed(futures):
            try:
                success = future.result()
                if success:
                    processed += 1
                    pbar.update(1)
            except Exception as e:
                logging.error(f"Error in parallel processing: {e}")

    pbar.close()
    logging.info(f"Processed {processed}/{total} files successfully")
    return processed
