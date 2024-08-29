import os
import torch
from faster_whisper import WhisperModel
import gradio as gr
from datetime import datetime
from moviepy.editor import VideoFileClip


def convert_video(folder):
    """
    Load in a video and extract the audio from it. Then, save the audio as mp3 and transcribe it.
    """
    start = datetime.now()
    for file in folder:
        # Save the temporary filename, extract the audio and save it
        _, original_name = os.path.split(file.name)
        filename, _ = os.path.splitext(original_name)

        # original_name = file.name.split('/')[-1]
        # file_name = file.name
        clip = VideoFileClip(file.name)

        # Define the name of the output file
        new_file = f"./audiodata/{filename}.mp3"
        clip.audio.write_audiofile(new_file, logger=None)
        # Remove the temporary video file after extracting the audio to clear memory
        del clip
        transcribe_save(new_file, filename)

    return (f'Done! {len(folder)} video files processed and created.\n'
            f'Saved in ./audioscribe/output/ folder!\n'
            f'Took {datetime.now()-start}')


def convert_audio(folder):
    """
    Load in an audio file and transcribe it.
    """
    start = datetime.now()
    for file in folder:
        # Get the filename without extension
        _, original_name = os.path.split(file.name)
        filename, _ = os.path.splitext(original_name)

        # Transcribe the audio file and save the output to a text file
        transcribe_save(file.name, filename)
    return (f'Done! {len(folder)} audio files processed and created.\n'
            f'Saved in ./audioscribe/output/ folder!\n'
            f'Took {datetime.now()-start}')


def transcribe_save(file_loc, filename):
    """
    Transcribe the audio file and save the output to a text file.
    """
    transcript, info = model.transcribe(file_loc)
    with open(f"./output/{filename}.txt", "w", encoding="utf-8") as output:
        for sentence in transcript:
            timestamp = f"[{round(sentence.start, 2)}s - {round(sentence.end, 2)}s]\n"
            text = sentence.text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n').lstrip()
            output.write(timestamp + text + '\n\n')


def initialize(model_size, use_gpu):
    """
    Function to choose and load a preferred model. Any GPU model selection is available, but the script will fall back
    to the next best option if the hardware does not support it.
    CPU models are restricted on RAM and will load regardless of GPU availability.
    """
    global model
    MEMORY = torch.cuda.mem_get_info()[-1] if torch.cuda.is_available() else 0
    try:
        if use_gpu:
            # Try the chosen option first
            if torch.cuda.is_available() and (MEMORY >= 3.8e9):
                model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                return f"Using the {model_size.upper()} model, running on GPU"
            elif torch.cuda.is_available() and (MEMORY >= 2.8e9):
                model = WhisperModel('medium', device="cuda", compute_type="int8_float32")
                return f"Using the MEDIUM model, running on GPU"
            elif torch.cuda.is_available() and (MEMORY >= 1.8e9):
                model = WhisperModel('small', device="cuda", compute_type="int8_float32")
                return f"Using the SMALL model, running on GPU"
            else:
                model = WhisperModel(model_size, device="cpu", compute_type="auto")
                return f"No GPU available, using the {model_size.upper()} model, running on CPU"
        else:
            model = WhisperModel(model_size, device="cpu", compute_type="auto")
            return f"Using the {model_size.upper()} model, running on CPU"

    except RuntimeError:
        print("An error has been encountered.")


def interface():
    """
    Create the Gradio interface for the user to interact with. The user can select the model size and whether to use the GPU.
    They can then upload audio or video files to transcribe. The output will be saved in the output folder.
    """
    # Select a model to use for transcription
    m_size = gr.Dropdown(label="Select model size", choices=["small", "medium", "large-v3"], value="large-v3", interactive=True)
    check = gr.Checkbox(label="Use GPU (if available)", value=True, interactive=True)
    init = gr.Interface(fn=initialize, inputs=[m_size, check], outputs=gr.Textbox(label="Selected model will be the best available option"), allow_flagging="never")

    # Transcribe audio files
    audio_upload = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Upload audio file(s)")
    audio = gr.Interface(fn=convert_audio, inputs=audio_upload, outputs=gr.Textbox(label="Output"), title="Transcribe Audio", allow_flagging="never")

    # Transcribe video files
    video_upload = gr.File(file_count="multiple", file_types=["video"], type="filepath", label="Upload video file(s)")
    video = gr.Interface(fn=convert_video, inputs=video_upload, outputs=gr.Textbox(label="Output"), title="Transcribe Video", allow_flagging="never")

    # Combine interfaces into one object and launch in browser
    gr.TabbedInterface([init, audio, video], ["Setup", "Audio", "Video"]).launch(inbrowser=True)


if __name__ == '__main__':
    global model
    os.makedirs('./audiodata', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    interface()
