
# A Self-Supervised Trajectory Similarity Measurement Method Based on Contrastive Predictive Coding

This repository contains the official implementation of **TS-CPC**, a self-supervised trajectory similarity measurement framework based on Contrastive Predictive Coding (CPC). TS-CPC is designed to handle complex trajectory datasets efficiently while delivering robust and accurate similarity measurements.

## Key Features
- Sliding-Window-based Linear Interpolation (SW-LI): Enhances trajectory features by preserving fine-grained details and improving model performance.
- Trajectory Detour Point Offset (TDPO): Improves robustness against real-world noise and drift.
- TME/SFEM: Extracts essential motion dynamics, such as speed, orientation, and angular velocity, for better understanding of trajectory dynamics.

## Key Results

### Datasets
Achieves state-of-the-art results on:
- **Grab-Posisi:** Achieved high retrieval performance across varying down-sampling and distortion rates.
- **GeoLife:** Demonstrated significant improvements in trajectory similarity measurement accuracy.

### Performance
Demonstrates advantages in trajectory similarity measurement, including:
- Improved representation quality: High-quality spatiotemporal representations for accurate trajectory comparisons.
- Computational efficiency: Optimized performance, reducing processing time while maintaining accuracy.
- Robustness: Maintains high performance under varying sampling rates, perturbations, and real-world trajectory variations.

---

## Version Information

This implementation has been developed and tested with the following environment and tools:
- **Operating System**: Ubuntu 20.04
- **Python Version**: 3.8
- **Deep Learning Framework**: PyTorch 1.13.1
- **CUDA Version**: 11.7
- **GPU**: NVIDIA A100

---

## License
This project is licensed under the MIT License.

