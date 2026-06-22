# ComfyUI ZiT Upscale

Custom ComfyUI nodes to upscale, refine, and edit existing images utilizing Z-Image Turbo. 

This repository includes the **"Z-Image polish"** node (technically named `ZiTUpscaleRefine`), which is frequently used in high-quality SDXL workflows.

## Features
- **Z-Image Polish (ZiTUpscaleRefine)**: An advanced node used for upscaling and quality enhancement.
- Includes a sample workflow (`SDXL_MaxQuality.json`) inside the `workflows` directory to help you get started.

## Installation

### Method 1: Using ComfyUI Manager (Recommended)
1. Open ComfyUI and click on **Manager**.
2. Click on **Install via Git URL**.
3. Paste the repository URL: `https://github.com/thevesperhouse-hub/comfyui-ZiT-Upscale`
4. Click **Install**. ComfyUI Manager will automatically download the nodes and install the required dependencies.
5. Restart ComfyUI.

### Method 2: Manual Installation
1. Navigate to your ComfyUI `custom_nodes` directory in your terminal:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/thevesperhouse-hub/comfyui-ZiT-Upscale.git
   ```
3. Navigate into the cloned folder:
   ```bash
   cd comfyui-ZiT-Upscale
   ```
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Restart ComfyUI.

## Dependencies
This package requires `mediapipe` and `transformers` (which are automatically handled if installed via ComfyUI Manager or by running the pip install command).
