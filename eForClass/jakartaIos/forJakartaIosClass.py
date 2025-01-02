import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm
from src.data_reader.dataset import RawDatasetSingaporeForClass
from src.model.model import CDCK2ForSingapore
import logging
import os
from datetime import datetime
import h5py
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up logger for logging information and errors
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# Set random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Feature extraction using trained CPC model
def extract_features(args, list_txt, logger):
    try:
        # Set device
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Load the model
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)

        # Remove 'module.' prefix if present (for multi-GPU model saving)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)  # Load the modified state dict
        model.eval()  # Set model to evaluation mode

        # Initialize dataset and data loader
        dataset = RawDatasetSingaporeForClass(args.raw_hdf5, list_txt, args.trajectory_window)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # List to store features and labels
        features = []
        labels = []

        # Extract features using the model encoder
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(data_loader, desc="Extracting features", unit="batch"):
                batch_data = batch_data.float().to(device)
                output_features = model.encoder(batch_data)  # Extract features from encoder
                features.append(output_features.view(output_features.size(0), -1).cpu().numpy())  # Flatten features
                labels.append(batch_labels.numpy())  # Append labels

        # Convert features and labels to numpy arrays
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Apply PCA to reduce feature dimensions
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        features = pca.fit_transform(features)

        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Log the shape of the feature vectors
        logger.info(f"Feature vectors shape: {features.shape}")
        
        # Return features and labels as tensors
        return torch.tensor(features).to(device), torch.tensor(labels).to(device)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return None, None

# Define classifier model (MLP)
class TrajectoryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TrajectoryClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add Dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # Output two classes: 0 or 1 for the traffic mode
        )

    def forward(self, x):
        return self.classifier(x)


# Main function to extract features, train classifier and evaluate
def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and extract feature vectors')
    parser.add_argument('--model-path', type=str, default='snapshot/jakartaIosForClass/cdc-2024-11-12_23_08_04-model_best.pth', help='Path to the trained model file')
    parser.add_argument('--raw-hdf5', type=str, default='dataset/grab_possi/grab_possi_Jakarta_all_new/ios/jakartaIosForClassifier.hdf5', help='Path to the raw HDF5 file')
    parser.add_argument('--list-file', type=str, default='dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassTest.txt', help='Path to the list file containing trajectory IDs and labels')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--trajectory-window', type=int, default=1024, help='Window length to sample from each trajectory')
    parser.add_argument('--timestep', type=int, default=16, help='Timestep for the model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    parser.add_argument('--log-dir', type=str, default='eForClass/jakartaIos/logs', help='Directory to save log files')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(args.log_dir)

    # Extract features using the CPC model
    features, labels = extract_features(args, args.list_file, logger)
    if features is None or labels is None:
        logger.error("Feature extraction failed.")
        return

    # Define classifier
    input_dim = features.size(1)  # Feature dimension extracted by CPC model
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    classifier = TrajectoryClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    # Train classifier
    classifier.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate classifier
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(features)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        logger.info(f"Classification Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()