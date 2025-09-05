# Audioscribe: Setup and Usage Guide

**Audioscribe** is a tool that uses [`m-bain/whisperX`](https://github.com/m-bain/whisperX) to perform automated audio and video transcriptions. It comes with a user interface built with [`Gradio`](https://gradio.app/).

This guide will help you get Audioscribe up and running on both Windows and MacOS. Follow the instructions relevant to your operating system.

## Getting Started
### 1. Download the Repository
- Download the Audioscribe repository as a ZIP file.
- Extract the contents to a directory named `/audioscribe-main`.


## Windows Setup
### 1.5. Using a GPU (optional, but recommended)
To perform faster inference, you need to run the transcription models on a GPU (if you have one). To do this, you will need to install CUDA. To do so, [Visual Studio 2019 Community](https://visualstudio.microsoft.com/downloads/) needs to be installed on your system first. Then download and install CUDA 12.4, which can be downloaded and installed through [this link](https://developer.nvidia.com/cuda-12-4-1-download-archive).

### 2. Install Python
- If you already have a Python version installed, make sure it is Python 3.9 or newer.
- If Python is not installed on your system, download and install Python 3.11.9 from [this link](https://www.python.org/downloads/release/python-3119/).
- During installation, check the box to **Add Python to PATH** if given the option.
- After the installation, disable the max path length if prompted.

### 3. Install Dependencies
- Open Command Prompt in the `/audioscribe-main` directory (you can do this by typing `cmd` in the address bar of File Explorer).
- Run the following command to install the required Python dependencies: ```pip3 install -r requirements.txt```
- Then, update `ctranslate2` by running the command: ```pip3 install ctranslate2==4.6.0```


## MacOS Setup (Version 10.9 or Later)

### 2. Install Python
- Make sure to install Python 3.11.9 from [this link](https://www.python.org/downloads/release/python-3119/).
- After installation, disable the max path length if possible.

### 3. Install Dependencies
- Open a Terminal in the `/audioscribe-main` directory.
- Run the following command to install the required Python dependencies: ```pip3 install -r requirements.txt```
- Then, update `ctranslate2` by running the command: ```pip3 install ctranslate2==4.6.0```


## Using Audioscribe

### 1. Run the Program
- To start Audioscribe, run the `audioscribe.py` file by opening a Command Prompt (Windows) or Terminal (MacOS) in the  `/audioscribe-main` directory and type `python audioscribe.py` (Windows) or `python3 audioscribe.py` (MacOS). This will run the script, which will then automatically create two folders:
  - `audiodata`: Where extracted audio from videos will be saved.
  - `output`: Where the transcripts (.txt files) will be stored.

### 2. Select a Model
- In the Gradio interface, choose whether to use your GPU (if available) for faster processing.
- Select a model and press the **Initialize Model** button.
- Wait for the model to load before navigating to the **Transcribe** page.

### 3. Transcribe Files
- After transcription, the resulting text file(s) will be saved in the `output` directory.
