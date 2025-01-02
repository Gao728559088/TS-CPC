import torch
import h5py
import random

def generate_detour_trajectory(
    trajectory, offset_distance=50.0, length_ratio=(0.2, 0.4), random_seed=None
):
    """
    Generate a detour trajectory by applying the Trajectory Detour Point Offset (TDPO) method.

    Args:
        trajectory (torch.Tensor): Tensor of shape (N, 2) representing trajectory points (x, y).
        offset_distance (float): Maximum offset distance (rho_m) for controlling the amplitude factor.
        length_ratio (tuple): Range (min, max) for randomly selecting the sub-trajectory length ratio.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        torch.Tensor: Detour trajectory with offsets applied to selected sub-trajectory points.
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Ensure trajectory has at least two points for meaningful operation
    n = trajectory.size(0)
    if n < 2:
        raise ValueError("Trajectory must contain at least two points.")

    # Step 1: Randomly select sub-trajectory
    sub_length = random.randint(int(n * length_ratio[0]), int(n * length_ratio[1]))
    start_idx = random.randint(0, n - sub_length)
    end_idx = start_idx + sub_length - 1

    sub_trajectory = trajectory[start_idx:end_idx + 1]

    # Step 2: Initialize detour points
    detour_points = sub_trajectory.clone()

    # Step 3: Apply position offsets
    for i in range(1, detour_points.size(0)):  # Start from the second point to calculate distances
        prev_point = detour_points[i - 1]
        current_point = detour_points[i]
        
        # Calculate the distance between consecutive points
        distance = torch.norm(current_point - prev_point)

        # Dynamically compute the amplitude factor gamma
        gamma = offset_distance / distance if distance > 0 else 0

        # Generate random offsets based on Gaussian distribution
        offset_x = gamma * torch.normal(mean=0.0, std=1.0, size=(1,))
        offset_y = gamma * torch.normal(mean=0.0, std=1.0, size=(1,))

        # Apply offsets to the current point
        detour_points[i, 0] += offset_x.item()
        detour_points[i, 1] += offset_y.item()

    # Step 4: Replace original points with detour points in the trajectory
    new_trajectory = trajectory.clone()
    new_trajectory[start_idx:end_idx + 1] = detour_points

    # Step 5: Return the new detour trajectory
    return new_trajectory


def process_hdf5_with_tdpo(input_file, offset_distance=50.0, length_ratio=(0.2, 0.4), random_seed=None):
    """
    Process HDF5 file by applying the TDPO method to each trajectory and saving the modified data back.

    Args:
        input_file (str): Path to the HDF5 file.
        offset_distance (float): Maximum offset distance (rho_m) for controlling the amplitude factor.
        length_ratio (tuple): Range (min, max) for randomly selecting the sub-trajectory length ratio.
        random_seed (int, optional): Seed for reproducibility.
    """
    with h5py.File(input_file, "r+") as hdf:
        for dataset_name in hdf.keys():
            print(f"Processing dataset: {dataset_name}")
            dataset = hdf[dataset_name][()]

            # Ensure dataset is compatible
            if dataset.ndim == 2 and dataset.shape[1] == 2:
                trajectory = torch.tensor(dataset)  # Convert to PyTorch tensor
                modified_trajectory = generate_detour_trajectory(
                    trajectory, offset_distance, length_ratio, random_seed
                )
                # Overwrite the dataset with the modified trajectory
                del hdf[dataset_name]  # Delete the old dataset
                hdf.create_dataset(dataset_name, data=modified_trajectory.numpy())  # Write the new dataset
            else:
                print(f"Skipping dataset '{dataset_name}' due to incompatible shape: {dataset.shape}")

    print("HDF5 processing completed.")


