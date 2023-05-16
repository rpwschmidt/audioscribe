import os
import torch
import whisper
import gradio as gr
from datetime import datetime
from moviepy.editor import VideoFileClip


def convert_video(folder):
    start = datetime.now()

    for file in folder:
        # Save the temporary filename, extract the audio and save it
        original_name = file.orig_name
        file_name = file.name
        clip = VideoFileClip(file_name)

        # Define the name of the output file
        new_file = f"./audiodata/{original_name.split('.')[0]}.mp3"
        clip.audio.write_audiofile(new_file, logger=None)
        transcribe_save(new_file, original_name)
    return f'Done! {len(folder)} video files processed and created.\nSaved in ./Whisper/output/ folder!\nTook {datetime.now()-start}'


def convert_audio(folder):
    start = datetime.now()
    for file in folder:
        transcribe_save(file.name, file.orig_name)
    return f'Done! {len(folder)} audio files processed and created.\nSaved in ./Whisper/output/ folder!\nTook {datetime.now()-start}'


def transcribe_save(filename, original_name):
    transcript = model.transcribe(filename, fp16=False)['text']
    try:
        with open(f'./output/{"_".join(original_name.split(".")[:-1])}.txt', 'w') as output:
            output.write(transcript.replace('. ', '.\n').replace('? ', '?\n'))
    except Exception as e:
        print("An exception occurred:", str(e))
        with open(f'./output/{original_name.split(".")[0].split(os.sep)[-1]}.txt', 'w') as output:
            output.write(transcript.replace('. ', '.\n').replace('? ', '?\n'))
        


def initialize(model_size, use_gpu):
    """
    Function to choose and load a preferred model. Any gpu model selection is available, but the script will fall back
    to the next best option if the hardware does not support it.
    Cpu models are unrestricted and will load regardless of gpu availability.
    """
    global model
    MEMORY = torch.cuda.mem_get_info()[-1] if torch.cuda.is_available() else 0
    mem_size = {'small': 2e9, 'medium': 5e9, 'large-v2': 10e9}

    try:
        if use_gpu:
            # Try the chosen option first
            if torch.cuda.is_available() and (MEMORY >= mem_size[model_size]):
                model = whisper.load_model(model_size, 'cuda')
                return f"Using the {model_size} model, running on gpu"

            # Go down the ladder to find the next best option
            elif torch.cuda.is_available() and (MEMORY >= mem_size['medium']):
                model = whisper.load_model('medium', 'cuda')
                return f"Using the 'medium' model, running on gpu"

            elif torch.cuda.is_available() and (MEMORY >= mem_size['small']):
                model = whisper.load_model('small', 'cuda')
                return f"Using the 'small' model, running on gpu"

            else:
                model = whisper.load_model(model_size, 'cpu')
                return f"No gpu available, using the {model_size} model, running on cpu"
        else:
            model = whisper.load_model(model_size, 'cpu')
            return f"Using the {model_size} model, running on cpu"

    except RuntimeError:
        print("An error has been encountered.")


def interface():
    # Select a model to use for transcription
    m_size = gr.Dropdown(label="Select model size", choices=["small", "medium", "large-v2"], value="large-v2", interactive=True)
    check = gr.Checkbox(label="Use GPU (if available)", value=True, interactive=True)
    init = gr.Interface(fn=initialize, inputs=[m_size, check], outputs=gr.Textbox(label="Selected model - will be the best option available given your choice"), allow_flagging="never")

    # Transcribe audio files
    audio_upload = gr.File(file_count="multiple", file_types=["audio"], type="file", label="Upload audio file(s)")
    audio = gr.Interface(fn=convert_audio, inputs=audio_upload, outputs=gr.Textbox(label="Output"), title="Transcribe Audio", allow_flagging="never")

    # Transcribe video files
    video_upload = gr.File(file_count="multiple", file_types=["video"], type="file", label="Upload video file(s)")
    video = gr.Interface(fn=convert_video, inputs=video_upload, outputs=gr.Textbox(label="Output"), title="Transcribe Video", allow_flagging="never")

    # Combine interfaces into one object and launch in browser
    gr.TabbedInterface([init, audio, video], ["Setup", "Audio", "Video"]).launch(inbrowser=True)


if __name__ == '__main__':
    global model
    os.makedirs('./audiodata', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    interface()
