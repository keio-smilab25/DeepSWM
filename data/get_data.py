# -*- coding: utf-8 -*-
import os
import requests
import sys
import numpy as np
import cv2
import json
import time
import h5py
from io import BytesIO
from datetime import datetime, timedelta
from astropy.io import fits
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from dateutil import tz

# ---------- Configuration ----------
AIA_WAVELENGTHS = [
    '0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'
]
SAVE_ROOT = 'data/images'  # PNG images for display
H5_SAVE_ROOT = './ml/datasets/all_data_hours'  # H5 files for ML inference
XRS_PATH = 'data/xrs.json'  # X-ray flux data
XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def fetch_and_process_aia_image(wavelength, dt):
    """
    Fetch an AIA FITS image for the given wavelength and datetime,
    downsample to 256x256, then crop, flip, and restore to 256x256.
    
    Args:
        wavelength (str): AIA wavelength (e.g., '0171', '0193')
        dt (datetime): Target datetime
    
    Returns:
        numpy.ndarray: Processed AIA image (256x256) or zeros if failed
    """
    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    # Select data source based on year
    if year >= 2023:
        url = f"https://sdo5.nascom.nasa.gov/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/" \
              f"AIA{ymd}_{hour}0000_{wavelength}.fits"
    else:
        url = f"https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/" \
              f"AIA{ymd}_{hour}00_{wavelength}.fits"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Load FITS file
        hdul = fits.open(BytesIO(resp.content))
        img = hdul[1].data.astype(np.float32)
        hdul.close()

        # Apply gaussian blur before downsampling
        blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)

        # Downsample to 256x256
        img256 = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_AREA)

        # Crop margins to remove artifacts
        v_crop, h_crop = 20, 15
        cropped = img256[v_crop:-v_crop, h_crop:-h_crop]

        # Flip vertically to match reference orientation
        cropped = cropped[::-1, :]

        # Resize back to 256x256
        zh = img256.shape[0] / cropped.shape[0]
        zw = img256.shape[1] / cropped.shape[1]
        img_fixed = zoom(cropped, (zh, zw), order=1)

        return img_fixed.astype(np.float32)

    except Exception as e:
        print(f"❌ AIA {wavelength} fetch/process failed: {e}")
        # Return black image if anything goes wrong
        return np.zeros((256, 256), dtype=np.float32)


def download_hmi_image(dt):
    """
    Download and process HMI magnetogram image.
    
    Args:
        dt (datetime): Target datetime
    
    Returns:
        numpy.ndarray: Processed HMI image (256x256) or zeros if failed
    """
    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    url = f"https://jsoc1.stanford.edu/data/hmi/images/{year}/{month}/{day}/{ymd}_{hour}0000_M_1k.jpg"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # Decode image from bytes
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Apply gaussian blur before downsampling
        blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)

        # Downsample to 256x256
        img256 = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_AREA)

        # Remove text area at bottom by overwriting with zeros
        text_h = 7
        img256[-text_h:, :] = 0

        return img256.astype(np.float32)

    except Exception as e:
        print(f"❌ HMI fetch/process failed: {e}")
        return np.zeros((256, 256), dtype=np.float32)


def save_png(image, path):
    """
    Save image as PNG file with directory creation.
    
    Args:
        image (numpy.ndarray): Image array to save
        path (str): File path to save the image
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def save_h5(aia_images, hmi_image, dt):
    """
    Save AIA and HMI images to HDF5 format for ML inference.
    
    Args:
        aia_images (list): List of AIA image arrays
        hmi_image (numpy.ndarray): HMI image array
        dt (datetime): Timestamp for the data
    """
    try:
        os.makedirs(H5_SAVE_ROOT, exist_ok=True)
        filename = dt.strftime("%Y%m%d_%H0000.h5")
        filepath = os.path.join(H5_SAVE_ROOT, filename)

        # Process AIA images (ensure grayscale and correct shape)
        aia_images_fixed = []
        for img in aia_images:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            aia_images_fixed.append(img.astype(np.float32).reshape(256, 256))

        # Process HMI image (ensure grayscale and correct shape)
        if hmi_image.ndim == 3 and hmi_image.shape[-1] == 3:
            hmi_image = cv2.cvtColor(hmi_image, cv2.COLOR_BGR2GRAY)
        hmi_image = hmi_image.astype(np.float32).reshape(256, 256)

        # Stack all images: 9 AIA channels + 1 HMI channel = 10 channels
        X = np.stack(aia_images_fixed + [hmi_image])  # shape: (10, 256, 256)
        timestamp = dt.strftime("%Y%m%d_%H0000").encode()

        # Save to HDF5 format
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("X", data=X)
            f.create_dataset("timestamp", data=timestamp)

        print(f"✅ H5 saved: {filepath}")
    except Exception as e:
        print(f"❌ H5 save failed: {e}")


def update_xrs_json(dt):
    """
    Update XRS (X-ray flux) data from NOAA SWPC API.
    
    Args:
        dt (datetime): Target datetime for data collection
    """
    try:
        response = requests.get(XRS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ XRS API fetch failed: {e}")
        return

    # Find maximum flux in the hour before dt
    max_flux = 0
    for item in data:
        # Only process 0.1-0.8nm energy range
        if item.get("energy") != "0.1-0.8nm":
            continue
            
        ts = item.get("time_tag")
        try:
            obs_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            obs_time = obs_time.replace(tzinfo=tz.tzutc())
        except:
            continue

        # Check if observation is within the target hour
        if dt - timedelta(hours=1) < obs_time <= dt:
            try:
                flux = float(item.get("flux", 0))
                max_flux = max(max_flux, flux)
            except:
                continue

    # Update JSON file with new data
    time_str = dt.strftime('%Y%m%d%H')
    try:
        if os.path.exists(XRS_PATH):
            with open(XRS_PATH, 'r') as f:
                xrs_data = json.load(f)
        else:
            xrs_data = {}
            
        xrs_data[time_str] = max_flux
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(XRS_PATH), exist_ok=True)
        
        with open(XRS_PATH, 'w') as f:
            json.dump(xrs_data, f, indent=2)
            
        print(f"✅ XRS updated: {time_str} → {max_flux}")
    except Exception as e:
        print(f"❌ xrs.json update failed: {e}")


def main():
    """
    Main function to fetch and process solar observation data.
    
    Command line usage:
        python data/get_data.py [MMDDHH]
        
    Arguments:
        MMDDHH: Optional datetime in format MMDDHH (month, day, hour)
                If not provided, uses current time - 30 minutes
    
    Outputs:
        - PNG images in data/images/ for visualization
        - H5 files in ml/datasets/all_data_hours/ for ML inference
        - XRS data updates in data/xrs.json
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            arg = sys.argv[1]
            dt = datetime.now(tz=tz.tzutc()).replace(minute=0, second=0, microsecond=0)
            year = dt.year  # Use current year
            month = int(arg[:2])  # First 2 characters as month
            day = int(arg[2:4])   # Next 2 characters as day
            hour = int(arg[4:6])  # Last 2 characters as hour
            dt = datetime(year, month, day, hour, tzinfo=tz.tzutc())
        except Exception as e:
            print(f"❌ Failed to process command line argument: {e}")
            print("Usage: python data/get_data.py [MMDDHH]")
            return
    else:
        # Default: current time - 30 minutes, rounded to nearest hour
        now_jst = datetime.now(tz=tz.gettz('Asia/Tokyo')) - timedelta(minutes=30)
        dt = now_jst.astimezone(tz=tz.tzutc()).replace(minute=0, second=0, microsecond=0)

    print(f"🚀 Starting data collection for: {dt.strftime('%Y-%m-%d %H:00 UTC')}")

    # Update XRS data first
    update_xrs_json(dt)
    update_xrs_json(dt - timedelta(hours=1))  # Also update previous hour

    # Check if files already exist
    time_str = dt.strftime('%H')
    date_str = dt.strftime('%m%d')
    h5_path = os.path.join(H5_SAVE_ROOT, dt.strftime("%Y%m%d_%H0000.h5"))
    
    # PNG file paths
    png_paths = [
        os.path.join(SAVE_ROOT, date_str, f"{time_str}_aia_{wl}.png")
        for wl in AIA_WAVELENGTHS
    ]
    png_paths.append(os.path.join(SAVE_ROOT, date_str, f"{time_str}_hmi.png"))

    # Skip if all files already exist
    if os.path.exists(h5_path) and all(os.path.exists(png) for png in png_paths):
        print(f"✅ All files already exist for {dt.strftime('%Y-%m-%d %H:00')}")
        return

    print("📡 Fetching solar observation data...")

    # Process AIA images for all wavelengths
    aia_images = []
    for wl in AIA_WAVELENGTHS:
        print(f"  Fetching AIA {wl}Å...")
        image_data = fetch_and_process_aia_image(wl, dt)
        aia_images.append(image_data)

        # Save as PNG for visualization (only if not completely black)
        if not np.all(image_data == 0):
            # Apply log scaling and normalization for display
            image_display = np.log1p(image_data)
            image_display = cv2.normalize(image_display, None, 0, 255, cv2.NORM_MINMAX)
            image_uint8 = image_display.astype(np.uint8)
            
            png_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_aia_{wl}.png")
            save_png(image_uint8, png_path)
            print(f"    ✅ PNG saved: {png_path}")
        else:
            print(f"    ⚠️  AIA {wl} data unavailable, skipping PNG")

    # Process HMI magnetogram
    print("  Fetching HMI magnetogram...")
    hmi_image = download_hmi_image(dt)
    
    # Save HMI PNG (only if not completely black)
    if not np.all(hmi_image == 0):
        hmi_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_hmi.png")
        save_png(hmi_image.astype(np.uint8), hmi_path)
        print(f"    ✅ PNG saved: {hmi_path}")
    else:
        print(f"    ⚠️  HMI data unavailable, skipping PNG")

    # Save H5 file for ML inference
    print("💾 Saving H5 file for ML inference...")
    save_h5(aia_images, hmi_image, dt)

    print(f"🎉 Data collection completed for {dt.strftime('%Y-%m-%d %H:00 UTC')}")


if __name__ == '__main__':
    main() 
