# Audioscribe: Setup and Usage Guide

**Audioscribe** is a tool that leverages [`openai/whisper`](https://github.com/openai/whisper.git) and [`SYSTRAN/faster-whisper`](https://github.com/SYSTRAN/faster-whisper) to perform automated audio and video transcriptions. It comes with a user interface built with [`Gradio`](https://gradio.app/).

This guide will help you get Audioscribe up and running on both Windows and MacOS. Follow the instructions relevant to your operating system.

## Getting Started
### 1. Download the Repository
- Download the Audioscribe repository as a ZIP file.
- Extract the contents to a directory named `/audioscribe-main`.

## Windows Setup
### 1.5. Using a GPU (optional)
If you want to run these models on a GPU (if you have one), which means faster inference, you will need to install CUDA. To install CUDA, [Visual Studio 2019 Community](https://visualstudio.microsoft.com/downloads/) needs to be installed on your system first. For this implementation of Whisper, you need CUDA 11.7, which can be downloaded and installed through [this link](https://developer.nvidia.com/cuda-11-7-1-download-archive).

### 2. Install Python
- If Python is not installed on your system, download and install Python 3.11.3 from [this link](https://www.python.org/ftp/python/3.11.3/python-3.11.3-amd64.exe).
- During installation, ensure you check the box to **Add Python to PATH**.
- After the installation, disable the max path length if prompted.

### 3. Install ffmpeg
Audioscribe requires the [`ffmpeg`](https://ffmpeg.org/) command-line tool for processing media files.

- To install `ffmpeg`, use the [`Chocolatey`](https://chocolatey.org/) package manager. 
- Open Command Prompt with administrative privileges (not the Python console) and run the following command:

```@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"```

- If you do not have administrative rights, please follow [these steps](https://docs.chocolatey.org/en-us/choco/setup#non-administrative-install) instead.

- After Chocolatey is installed, install `ffmpeg` by running:

  ```choco install ffmpeg```

- When prompted "Do you want to run the script? ([Y]es/[A]ll/[N]o/[P]rint):", type `Y` and press Enter.

### 4. Install Git
- If Git is not installed, download and install the latest version from [here](https://git-scm.com/download/win).

### 5. Install Dependencies
- Open Command Prompt in the `/audioscribe-main` directory (you can do this by typing `cmd` in the address bar of File Explorer).
- Run the following commands to install the required Python dependencies:

    - If you installed CUDA first:

      ```pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117```

    - If you did not install CUDA:

      ```pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2```

    - Followed by:
  
      ```pip3 install -r requirements.txt```

## MacOS Setup (Version 10.9 or Later)

### 2. Install Python
- If Python is not installed, download and install Python 3.11.3 from [this link](https://www.python.org/ftp/python/3.11.3/python-3.11.3-macos11.pkg).
- After installation, disable the max path length if possible.

### 3. Install ffmpeg
Audioscribe requires the [`ffmpeg`](https://ffmpeg.org/) command-line tool for processing media files.

- Open Terminal and install [`Homebrew`](https://brew.sh/) by running:

  ```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```

- Once Homebrew is installed, install `ffmpeg` by running:

  ```brew install ffmpeg```

### 4. Install Git
- If Git is not installed, open Terminal and run:

  ```brew install git```

### 5. Install Dependencies
- Open Terminal in the `/audioscribe-main` directory.
  
- Run the following commands to install the required Python dependencies:
  
  ```pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2```
  
- Followed by:
  
  ```pip3 install -r requirements.txt```

## Using Audioscribe

### 1. Run the Program
- To start Audioscribe, run the `audioscribe.py` file by opening a Command Prompt (Windows) or Terminal (MacOS) in the  `/audioscribe-main` directory and type `python audioscribe.py`. This will run the script and automatically create two folders:
  - `audiodata`: Where extracted audio from videos will be saved.
  - `output`: Where the transcriptions (.txt files) will be stored.

### 2. Select a Model
- In the Gradio interface, select a model and press the **Submit** button.
    - As a general rule of thumb, smaller models are faster but less accurate, while larger models are slower but more accurate.
- Wait for the model to load before navigating to the **Audio** or **Video** pages.

### 3. Transcribe Media Files
- Once the model is loaded, you can start transcribing audio or video files. The resulting text files will be saved in the `output` directory.


## FAQ
1. When trying to load a model, I get the following error: `Exception in ASGI application`
  - Solution: Make sure to uncomment the last two lines in requirements.txt, and rerun the install command. This downgrades two modules, which solves the error. Alternatively, run the following command: `pip install --upgrade pydantic==2.8.0 fastapi==0.112.4`.
 
