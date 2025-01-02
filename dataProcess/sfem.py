import torch
import h5py

def compute_shallow_features(trajectory, timestamps):
    """
    Compute shallow motion features for a given trajectory.

    Args:
        trajectory (torch.Tensor): Tensor of shape (N, 2), where each row is a point (latitude, longitude).
        timestamps (torch.Tensor): Tensor of shape (N,), where each element is a timestamp.

    Returns:
        dict: A dictionary containing extracted shallow features:
              - speed
              - speed_variation
              - orientation
              - orientation_variation
              - angular_velocity
              - curvature
    """
    assert trajectory.size(0) == timestamps.size(0), "Trajectory and timestamps must have the same length."

    features = {}
    delta_pos = trajectory[1:] - trajectory[:-1]
    distances = torch.norm(delta_pos, dim=1)
    delta_time = timestamps[1:] - timestamps[:-1]

    # Step 1: Speed
    speed = distances / delta_time
    features["speed"] = speed

    # Step 2: Speed Variation
    speed_variation = speed[1:] - speed[:-1]
    features["speed_variation"] = speed_variation

    # Step 3: Orientation
    orientation = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
    features["orientation"] = orientation

    # Step 4: Orientation Variation
    orientation_variation = orientation[1:] - orientation[:-1]
    features["orientation_variation"] = orientation_variation

    # Step 5: Angular Velocity
    angular_velocity = orientation_variation / delta_time[1:]
    features["angular_velocity"] = angular_velocity

    # Step 6: Curvature
    curvature = []
    for i in range(trajectory.size(0) - 2):
        p1, p2, p3 = trajectory[i], trajectory[i + 1], trajectory[i + 2]
        A = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        dq = torch.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        if dq != 0:
            kappa = 2 * abs(A) / dq**3
        else:
            kappa = 0
        curvature.append(kappa)
    features["curvature"] = torch.tensor(curvature)

    return features


def process_hdf5_file_with_features(input_file, output_file):
    """
    Read trajectory data from an HDF5 file, compute shallow motion features, 
    and write the updated data into a new HDF5 file.

    Args:
        input_file (str): Path to the input HDF5 file containing raw trajectory data.
        output_file (str): Path to the output HDF5 file where results will be saved.
    """
    with h5py.File(input_file, "r") as hdf_in, h5py.File(output_file, "w") as hdf_out:
        for dataset_name in hdf_in.keys():
            print(f"Processing dataset: {dataset_name}")
            data = hdf_in[dataset_name][()]
            
            # Extract latitude, longitude, and timestamps
            trajectory = torch.tensor(data[:, :2])  # First two columns are latitude and longitude
            timestamps = torch.tensor(data[:, 2])  # Third column is timestamps

            # Compute shallow features
            features = compute_shallow_features(trajectory, timestamps)

            # Prepare new dataset
            num_points = trajectory.size(0)
            new_data = torch.zeros((num_points, 9))  # Original 3 columns + 6 new features
            new_data[:, :2] = trajectory
            new_data[:, 2] = timestamps

            # Fill in shallow features
            new_data[1:, 3] = features["speed"]
            new_data[2:, 4] = features["speed_variation"]
            new_data[1:, 5] = features["orientation"]
            new_data[2:, 6] = features["orientation_variation"]
            new_data[2:, 7] = features["angular_velocity"]
            new_data[2:, 8] = torch.cat((features["curvature"], torch.zeros(2)))  # Padding curvature

            # Write to new HDF5 file
            hdf_out.create_dataset(dataset_name, data=new_data.numpy())

    print(f"Processing completed. Results saved to {output_file}")


