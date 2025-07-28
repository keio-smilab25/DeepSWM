import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from typing import Optional, Tuple
from torch.utils.data.dataloader import DataLoader


def visualize_model_outputs(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    trial_name: str,
    num_images: int = 30,
    use_sunspot_masking: bool = True,
    time_range: Optional[Tuple] = None,
) -> bool:
    """
    Visualize model outputs

    Parameters:
        model: Model to visualize
        data_loader: Data loader
        device: Device
        output_dir: Output directory
        trial_name: Trial name
        num_images: Number of images to visualize
        use_sunspot_masking: Whether to use sunspot masking
        time_range: Time range

    Returns:
        success: Whether the processing was successful
    """
    try:
        visualize_reconstruction(
            model,
            data_loader,
            device,
            output_dir,
            trial_name,
            num_images=num_images,
            use_sunspot_masking=use_sunspot_masking,
            time_range=time_range,
        )
        return True
    except Exception as e:
        print(f"Error visualizing model outputs: {str(e)}")
        return False


def visualize_reconstruction(
    model,
    test_loader,
    device,
    output_dir,
    trial_name,
    num_images=30,
    use_sunspot_masking=False,
    time_range=None,
):
    """
    Visualization of reconstructed images

    Parameters:
      model: Model to use
      test_loader: Test data DataLoader (batch_size=1 recommended)
      device: Device to use
      output_dir: Output directory for result images
      trial_name: Trial name
      num_images: Number of samples to display
      use_sunspot_masking: Whether to use sunspot region masking
      time_range: Time range to select (e.g., (start_time, end_time))
                  Note: start_time, end_time should be string or datetime type
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Convert time_range to pd.Timestamp if provided
    if time_range is not None:
        start_time = pd.to_datetime(time_range[0])
        end_time = pd.to_datetime(time_range[1])
        time_range = (start_time, end_time)
        print(f"DEBUG: Time range to use → {time_range[0]} to {time_range[1]}")

    selected_samples = []
    total_scanned = 0
    not_matching_count = 0

    # Since batch size is 1, timestamp returned by __getitem__ is a list with 1 element
    for images, _, timestamp in tqdm(test_loader, desc="Processing samples"):
        total_scanned += 1
        ts = timestamp[0]
        if not isinstance(ts, pd.Timestamp):
            try:
                ts = pd.to_datetime(ts)
            except Exception as e:
                print(f"DEBUG: Cannot convert timestamp: {ts} → {e}")
                continue
        print(f"DEBUG: sample timestamp = {ts}, time_range = {time_range}")
        if time_range is not None:
            if not (time_range[0] <= ts <= time_range[1]):
                not_matching_count += 1
                if not_matching_count <= 5:
                    print(
                        f"DEBUG: This sample does not match the filter: {ts} not in {time_range[0]} to {time_range[1]}"
                    )
                continue
        else:
            if random.random() > 0.1:
                continue
        selected_samples.append((images, ts))
        if len(selected_samples) >= num_images:
            break

    print(
        f"DEBUG: Number of samples scanned = {total_scanned}, Number not matching filter = {not_matching_count}"
    )
    print(f"DEBUG: Number of samples selected = {len(selected_samples)}")
    if len(selected_samples) == 0:
        print("WARNING: No samples found matching the specified time range.")
        print("Some available samples:")
        for images, _, timestamp in test_loader:
            ts = timestamp[0]
            if not isinstance(ts, pd.Timestamp):
                ts = pd.to_datetime(ts)
            print(f"   {ts}")
            total_scanned += 1
            if total_scanned >= 5:
                break
        return

    print(f"Processing {len(selected_samples)} selected samples...")

    # Process each selected sample
    for idx, (images, ts) in enumerate(selected_samples):
        images = images.to(device)

        # Get mask images and reconstructed images from the model
        if use_sunspot_masking:
            latent, mask, ids_restore = model.forward_encoder_pyramid(
                images,
                mask_ratio=0.5,
                sunspot_spatial_ratio=0.3,
                feature_mask_ratio=0.25,
            )
            pred = model.forward_decoder(latent, ids_restore)
            num_patches = int((images.shape[2] / model.patch_embed.patch_size[0]) ** 2)
            mask = mask.reshape(mask.shape[0], num_patches)
            mask = mask.unsqueeze(-1).repeat(
                1, 1, model.patch_embed.patch_size[0] ** 2 * 10
            )
            mask_img = model.unpatchify_dim10(mask)
            recon_img = model.unpatchify_dim10(pred)
        else:
            reconstructed, mask = model(images)
            num_patches = int((images.shape[2] / model.patch_embed.patch_size[0]) ** 2)
            mask = mask.reshape(mask.shape[0], num_patches)
            mask = mask.unsqueeze(-1).repeat(
                1, 1, model.patch_embed.patch_size[0] ** 2 * 10
            )
            mask_img = model.unpatchify_dim10(mask)
            recon_img = model.unpatchify_dim10(reconstructed)

        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")

        # Create folder for each sample
        sample_dir = os.path.join(output_dir, f"sample_{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 4500]

        # Process each channel (assuming 10 channels total)
        for ch in range(10):
            # Create subfolder for channel
            channel_dir = os.path.join(sample_dir, f"channel_{ch}")
            os.makedirs(channel_dir, exist_ok=True)

            # Select title and colormap based on channel
            if ch < 9:
                wavelength = wavelengths[ch]
                title_prefix = f"Channel {ch}: {wavelength}Å\n{timestamp_str}"
                cmap_used = "hot"
            else:
                title_prefix = f"Channel {ch}: HMI\n{timestamp_str}"
                cmap_used = "viridis"

            # Get original image
            orig_img = images[0, ch].cpu().numpy()

            # Generate standard deviation map (calculated per patch for each channel)
            H_img, W_img = orig_img.shape
            patch_size = model.patch_embed.patch_size[0]
            std_map = np.zeros((H_img, W_img))
            for i in range(0, H_img - patch_size + 1, patch_size):
                for j in range(0, W_img - patch_size + 1, patch_size):
                    patch = (
                        images[0, ch, i : i + patch_size, j : j + patch_size]
                        .cpu()
                        .numpy()
                    )
                    std_map[i : i + patch_size, j : j + patch_size] = np.std(patch)
            valid_stds = std_map[std_map > 0]
            std_threshold = np.quantile(valid_stds, 0.80) if valid_stds.size > 0 else 0
            high_std_mask = std_map > std_threshold

            # STD map display image: overlay transparent white layer on non-masked areas
            alpha_std = (
                0.3  # transparency (0: original image only, 1: completely white)
            )
            std_display = np.where(
                high_std_mask,
                orig_img,
                (1 - alpha_std) * orig_img + alpha_std * np.ones_like(orig_img),
            )

            # チャネルごとにマスクを生成（標準偏差の上位20%と下位80%で異なるマスク率を適用）
            channel_mask = torch.zeros_like(images[0, ch]).to(device)
            for i in range(0, H_img - patch_size + 1, patch_size):
                for j in range(0, W_img - patch_size + 1, patch_size):
                    patch_std = std_map[i : i + patch_size, j : j + patch_size].mean()
                    if patch_std > std_threshold:
                        # 標準偏差上位20%の領域には0.3のマスク率を適用
                        if random.random() < 0.3:
                            channel_mask[i : i + patch_size, j : j + patch_size] = 1.0
                    else:
                        # 標準偏差下位80%の領域には0.75のマスク率を適用
                        if random.random() < 0.75:
                            channel_mask[i : i + patch_size, j : j + patch_size] = 1.0

            # チャネルごとのマスクを表示用に変換
            channel_mask_np = channel_mask.cpu().numpy()
            alpha_mask = 0.5
            masked_display = np.where(
                channel_mask_np > 0.5,
                (1 - alpha_mask) * orig_img + alpha_mask * np.ones_like(orig_img),
                orig_img,
            )
            plt.imsave(
                os.path.join(channel_dir, "masked_regions.png"),
                masked_display,
                cmap=cmap_used,
            )

            # Reconstructed image
            recon_display = recon_img[0, ch].detach().cpu().numpy()

            # Combine 4 images into one figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for ax in axes:
                ax.axis("off")

            axes[0].imshow(orig_img, cmap=cmap_used)
            axes[0].set_title(f"{title_prefix}\nOriginal", fontsize=12)

            axes[1].imshow(std_display, cmap=cmap_used)
            axes[1].set_title(f"{title_prefix}\nSTD Map", fontsize=12)

            axes[2].imshow(masked_display, cmap=cmap_used)
            axes[2].set_title(f"{title_prefix}\nMasked Regions", fontsize=12)

            axes[3].imshow(recon_display, cmap=cmap_used)
            axes[3].set_title(f"{title_prefix}\nReconstructed", fontsize=12)

            plt.tight_layout()

            # Save the combined image (combined.png)
            combined_path = os.path.join(channel_dir, "combined.png")
            plt.savefig(combined_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Save individual images
            plt.imsave(
                os.path.join(channel_dir, "original.png"), orig_img, cmap=cmap_used
            )
            plt.imsave(
                os.path.join(channel_dir, "std.png"), std_display, cmap=cmap_used
            )

            print(f"Saved channel {ch} visualizations in {channel_dir}")

        print(f"Saved visualizations for sample {idx} with timestamp: {timestamp_str}")

    print("Visualization complete.")
