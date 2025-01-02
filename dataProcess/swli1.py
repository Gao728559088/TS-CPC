import torch
import h5py

def sliding_window_linear_interpolation_pytorch(
    trajectory, target_points, window_size=4, step_size=3, theta_threshold=0.1
):
    """
    Perform Sliding-Window-based Linear Interpolation (SW-LI) using only PyTorch.

    Args:
        trajectory (torch.Tensor): Tensor of shape (N, 2) representing trajectory points (x, y).
        target_points (int): Target number of trajectory points.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        theta_threshold (float): Threshold for average azimuth change to determine interpolation.

    Returns:
        torch.Tensor: Interpolated trajectory.
    """
    current_points = trajectory.size(0)
    if current_points >= target_points:
        return trajectory  # No interpolation required if current points meet or exceed target

    # Clone the trajectory for safe operations
    interpolated_trajectory = trajectory.clone()
    
    while interpolated_trajectory.size(0) < target_points:
        new_points = []

        # Generate sliding windows
        for i in range(0, interpolated_trajectory.size(0) - window_size + 1, step_size):
            # Get current window points
            window_points = interpolated_trajectory[i:i + window_size]
            
            # Calculate azimuth differences
            dx = window_points[1:, 0] - window_points[:-1, 0]  # delta x
            dy = window_points[1:, 1] - window_points[:-1, 1]  # delta y
            deltas = torch.atan2(dy, dx)  # Azimuth changes
            avg_delta = deltas.mean().abs()  # Average azimuth change

            # Check if interpolation is required
            if avg_delta < theta_threshold:
                midpoint = 0.5 * (window_points[1] + window_points[0])  # Midpoint between first two points
                new_points.append((i + 1, midpoint))  # Collect the new point

        # Insert new points
        for idx, new_point in reversed(new_points):
            interpolated_trajectory = torch.cat(
                [interpolated_trajectory[:idx], new_point.unsqueeze(0), interpolated_trajectory[idx:]], dim=0
            )
        
        # Stop if target points are reached
        if interpolated_trajectory.size(0) >= target_points:
            break

    # Trim if trajectory exceeds the target points
    if interpolated_trajectory.size(0) > target_points:
        interpolated_trajectory = interpolated_trajectory[:target_points]

    return interpolated_trajectory


def process_hdf5_file(input_file, target_points, window_size=4, step_size=3, theta_threshold=0.1):
    """
    Process trajectories stored in an HDF5 file using sliding window interpolation
    and save the results back to the same file.

    Args:
        input_file (str): Path to the input HDF5 file.
        target_points (int): Target number of trajectory points.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        theta_threshold (float): Threshold for average azimuth change to determine interpolation.
    """
    with h5py.File(input_file, "r+") as hdf:
        for dataset_name in hdf.keys():
            print(f"Processing dataset: {dataset_name}")
            dataset = hdf[dataset_name][()]
            
            # Ensure the dataset is compatible
            if dataset.ndim == 2 and dataset.shape[1] == 2:
                trajectory = torch.tensor(dataset)  # Convert to PyTorch tensor
                interpolated_trajectory = sliding_window_linear_interpolation_pytorch(
                    trajectory, target_points, window_size, step_size, theta_threshold
                )
                # Write the interpolated trajectory back to the dataset
                del hdf[dataset_name]  # Remove old dataset
                hdf.create_dataset(dataset_name, data=interpolated_trajectory.numpy())  # Write new dataset
            else:
                print(f"Skipping dataset '{dataset_name}' due to incompatible shape: {dataset.shape}")

    print("HDF5 processing completed.")


