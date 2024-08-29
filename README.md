# audioscribe
Audioscribe is an implementation of [`openai/whisper`](https://github.com/openai/whisper.git) and [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) that can perform automated audio and video transcriptions. A [`Gradio`](https://gradio.app/) interface is built on top.

The following text is a manual to help you get Audioscribe up and running. 
Please follow the instructions for Windows or MacOS, depending on your operating system.
First, download this repository as a zip file and extract this to `/audioscribe-main`

### Windows
If Python has not yet been installed, please do so first. Python 3.11.3 can be downloaded through [this link](https://www.python.org/ftp/python/3.11.3/python-3.11.3-amd64.exe). When installing Python, make sure to check the box to `Add Python to PATH`. After the installation is finished, disable the max path length if possible.

For Audioscribe to function, the [`ffmpeg CLI`](https://ffmpeg.org/) needs to be installed. To do this, we can use [`chocolatey`](https://chocolatey.org/). Run the following command in the Command Prompt terminal with administrator rights (CMD, not the Python console). If you do not have administrative rights, please follow [these steps](https://docs.chocolatey.org/en-us/choco/setup#non-administrative-install) instead.

`@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"`

After the installation process has finished, run the following command: `choco install ffmpeg`. When prompted 'Do you want to run the script?([Y]es/[A]ll - yes to all/[N]o/[P]rint):' type in `Y` and press `Enter`.

To install `openai/whisper` we need to have `git` installed. Please download and install the [latest maintained build](https://git-scm.com/download/win) of Git.

Finally, open a Command Prompt terminal in the `./audioscribe-main` folder (this can be done by typing 'cmd' in the address bar of the File Explorer) and run the following command to install the required dependencies: `pip install -r requirements.txt`.


### MacOS >= 10.9
If Python has not yet been installed, please do so first. You can download and install Python 3.11.3 through [this link](https://www.python.org/ftp/python/3.11.3/python-3.11.3-macos11.pkg). After the installation is finished, disable the max path length if possible. 

For Audioscribe to function, the [`ffmpeg CLI`](https://ffmpeg.org/) needs to be installed. To do this, [`brew`](https://brew.sh/) can be used. Open a new Terminal and run the following command: 

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

After the installation process is finished, type or copy in the following command and follow the instructions when prompted: `brew install ffmpeg`.

To install `openai/whisper` we need to have `git` installed. Please download and install Git by running `brew install git` in the Terminal.

Finally, open a Terminal in the `./audioscribe-main` folder and run the following command to install the required dependencies: `pip install -r requirements.txt`.

### Using Audioscribe
Running the `audioscribe.py` file will start the program and create two folders: `audiodata` and `output`. When transcribing videos, the extracted audio will be saved in `./audiofolder`. All output `.txt` files will be placed in `./output`. When using Audioscribe, please select a model and press the "Submit" button. Please wait until the model is loaded before moving to the "Audio" or "Video" page.
