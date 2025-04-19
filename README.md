# ESM_ICP

**ESM_ICP** is a C++ implementation of the Enhanced Soft Matching Iterative Closest Point (ESM-ICP) algorithm for point cloud registration. This repository provides tools for aligning 3D point clouds using the ESM-ICP method.

<video width="640" height="360" autoplay loop muted playsinline>
  <source src="output1.gif" type="image/gif">
</video>

<video width="640" height="360" autoplay loop muted playsinline>
  <source src="output2.gif" type="image/gif">
</video>

<video width="640" height="360" autoplay loop muted playsinline>
  <source src="output3.gif" type="image/gif">
</video>

<video width="640" height="360" autoplay loop muted playsinline>
  <source src="output4.gif" type="image/gif">
</video>

## Features

- Implementation of the ESM-ICP algorithm for precise point cloud registration.
- Visualization tools for analyzing registration results.
- Modular code structure for easy integration and extension.

## Getting Started

### Prerequisites

- C++17 compatible compiler
- [Eigen](https://eigen.tuxfamily.org/) library
- [PCL (Point Cloud Library)](https://pointclouds.org/) (optional, for visualization)

### Building the Project

1. Clone the repository:

   ```bash
   git clone --recurse-submodules https://github.com/aralab-unr/ESM_ICP.git
   cd ESM_ICP
