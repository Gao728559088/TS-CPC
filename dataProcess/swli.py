import torch

def sliding_window_linear_interpolation(trajectory, target_points, window_size=4, step_size=3, theta_threshold=0.1):
    """
    Perform Sliding-Window-based Linear Interpolation (SW-LI) on trajectory data.

    Args:
        trajectory (torch.Tensor): Tensor of shape (N, 2) representing the trajectory points (x, y).
        target_points (int): Target number of trajectory points.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        theta_threshold (float): Threshold for average azimuth change to determine interpolation.

    Returns:
        torch.Tensor: Interpolated trajectory.
    """
    current_points = trajectory.size(0)
    points_to_insert = target_points - current_points
    
    if points_to_insert <= 0:
        return trajectory  # No interpolation required

    interpolated_trajectory = trajectory.clone()
    
    while interpolated_trajectory.size(0) < target_points:
        new_points = []
        for i in range(0, interpolated_trajectory.size(0) - window_size + 1, step_size):
            # Extract points in the current window
            window_points = interpolated_trajectory[i:i + window_size]
            # Calculate azimuth changes
            deltas = torch.atan2(
                window_points[1:, 1] - window_points[:-1, 1],
                window_points[1:, 0] - window_points[:-1, 0]
            )
            avg_delta = deltas.mean().abs()
            # Check if interpolation condition is met
            if avg_delta < theta_threshold:
                midpoint = 0.5 * (window_points[1] + window_points[0])
                new_points.append((i + 1, midpoint))
        
        # Insert new points
        for idx, new_point in reversed(new_points):
            interpolated_trajectory = torch.cat(
                [interpolated_trajectory[:idx], new_point.unsqueeze(0), interpolated_trajectory[idx:]], dim=0
            )
        
        # Stop if the target is reached
        if interpolated_trajectory.size(0) >= target_points:
            break
    
    # Trim if we exceed the target
    if interpolated_trajectory.size(0) > target_points:
        interpolated_trajectory = interpolated_trajectory[:target_points]
    
    return interpolated_trajectory



