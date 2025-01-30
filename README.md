# ISIC-2018-Task3-Segmentation

## Repository Description
This repository contains the code and supplementary materials used for generating automatic segmentation masks for the **ISIC-2018 Challenge Task 3 Test Set** using a **UNet-based pipeline**. The dataset and code are designed to support further research in **skin lesion segmentation, deep learning, and medical image analysis**.

## Reference Paper & Dataset
The work is associated with the dataset:
**Brzozowski, K. P., Bobowicz, M., Buler, J., Buler, R., & Grochowski, M. (2025).** *ISIC-2018 Task 3 Test Set (Auto-Segmentation Masks) [Dataset]. Gdańsk University of Technology.* [DOI: 10.34808/j8dh-2049](https://doi.org/10.34808/j8dh-2049)

## Repository Contents
- **Preprocessing scripts** for ISIC-2018 Task 3 images
- **UNet model implementation** for generating segmentation masks
- **Post-processing techniques** for lesion boundary refinement

---

## Running the Project with Docker Compose
### **Prerequisites**
Before running the project, make sure you have:
- **Docker** installed ([Install Docker](https://docs.docker.com/get-docker/))
- **NVIDIA drivers** (if using GPU acceleration)
- **Docker Compose** installed ([Install Docker Compose](https://docs.docker.com/compose/install/))

### **Setup Environment Variables**
This project uses environment variables to configure paths for project files and datasets. To set them up:

1. Copy the example environment file:
   ```sh
   cp .env.example .env
   ```
2. Edit `.env` to define your local paths:
   ```ini
   PROJECT_PATH=/absolute/path/to/your/project
   DATA_PATH=/absolute/path/to/your/dataset
   ```

### **Building & Running the Docker Container**
1. **Build the Docker image**:
   ```sh
   docker-compose build
   ```

2. **Run the container**:
   ```sh
   docker-compose up
   ```
   This will start the container and execute `main.py` automatically.

3. **Access the running container (optional)**:
   If you need to manually interact with the container:
   ```sh
   docker exec -it lesion_segmentation_container /bin/bash
   ```

4. **Stopping the container**:
   To stop the running container, press `Ctrl+C` or run:
   ```sh
   docker-compose down
   ```

---

## **Using GPU Acceleration**
This project supports **GPU acceleration** with NVIDIA. Ensure your system meets the following requirements:
- **NVIDIA Container Toolkit** installed ([Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Run the container with GPU support enabled automatically via Docker Compose.

---

## **Citation**
If you use this code or dataset in your research, please cite the related dataset and acknowledge **EUCAIM and ISIC-2018 Challenge** as sources.

```bibtex
@dataset{Brzozowski2025ISIC,
  author    = {Krystian P. Brzozowski and Maciej Bobowicz and Jakub Buler and Rafał Buler and Michał Grochowski},
  title     = {ISIC-2018 Task 3 Test Set (Auto-Segmentation Masks)},
  year      = {2025},
  publisher = {Gdańsk University of Technology},
  doi       = {10.34808/j8dh-2049},
  url       = {https://doi.org/10.34808/j8dh-2049}
}
```

