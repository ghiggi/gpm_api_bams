# GPM-API BAMS Figures

This repository contains all the scripts needed to recreate every figure from the Bulletin of the American Meteorological Society article:

> **Dive into Global Precipitation Measurement Mission Data Without Getting Soaked:  
> How GPM‑API Helps You Stay Dry and Wise**

## 📂 Repository Structure

Each directory contains the scripts needed to reproduce a specific figure. When multiple scripts appear in a directory, a numeric prefix in the filename determines their execution order. Before running them, update the source and destination directory paths at the top of each file.

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourorg/gpm-api-figures.git
   cd gpm-api-figures
  
2. **Install the dependencies**
   ```bash
   conda env create -f environment.yaml
   
3. **Set up GPM-API**
  Before running the scripts, you have to configure the GPM-API software.
  To do so, please follow the [Quick Start Tutorial] (https://gpm-api.readthedocs.io/en/latest/03_quickstart.html).  

4. **Download GPM data**
 
  Reproducing Figure 6 and B1 involves downloading 4 GPM product archives: **1C-GMI-R**, **2A-GMI-CLIM**, **2A-DPR**  and **IMERG‑FR**  
   Downloading these archives will require **>100 TB** in disk space.
   Follow the instructions in the [GPM‑API Quick Start Guide](
   https://gpm-api.readthedocs.io/en/latest/03_quickstart.html#download-the-data)
   to download GPM data from the terminal or within a Python session.

   In terms of disk storage, please consider that the rechunking of IMERG data (to recreate FigB1) and the satellite bucketing of GMI and DPR products (required to reproduce Fig6) require an additional ~30 TB of disk space and several days of computation.

📖 Article Reference

Ghiggi, G., Pham-Ba, S., Berne, A. (2025). Dive into Global Precipitation Measurement Mission Data Without Getting Soaked: How GPM‑API Helps You Stay Dry and Wise.
Bulletin of the American Meteorological Society, XX(YY), pp. ZZ–AA. DOI
 