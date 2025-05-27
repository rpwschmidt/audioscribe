import gc
import torch
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
from moviepy import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel


class Audioscribe:
    """Main Audioscribe class to handle the transcription of audio and video files."""

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.audio_formats: Set[str] = {".mp3", ".aac", ".m4a", ".wav", ".flac", ".opus"}
        self.video_formats: Set[str] = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".mpeg", ".m4v", ".ogv"}
        self.gpu_memory_requirements = [
            (3.18e9, "large-v3"),
            (1.70e9, "large-v3-turbo"),
            (1.60e9, "medium"),
            (5.55e8, "small")
        ]
        self._setup_directories()
        self._device_setup()

    @staticmethod
    def print_info(text: str, message_type: str = "info") -> None:
        print(text)
        if message_type == "error":
            gr.Error(text)
        elif message_type == "warning":
            gr.Warning(text)
        else:
            gr.Info(text)

    @staticmethod
    def _setup_directories() -> None:
        """Create necessary directories for output."""
        Path("./audiodata").mkdir(exist_ok=True)
        Path("./output").mkdir(exist_ok=True)

    def _device_setup(self) -> None:
        """Setup device and memory information"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            if torch.cuda.is_available():
                self.memory = torch.cuda.get_device_properties(0).total_memory
            else:
                self.memory = 0
        except Exception as e:
            print(f"Error getting device properties: {e}")
            self.memory = 0

    def _select_model(self, model_size: str, use_gpu: bool) -> str:
        """Select and load a model based on choice, memory size, and GPU availability."""
        if not use_gpu or not torch.cuda.is_available():
            return model_size

        for mem_threshold, model_option in self.gpu_memory_requirements:
            if self.memory >= mem_threshold:
                # Return chosen model size if it meets the memory requirements
                size_hierarchy = ["small", "medium", "large-v3-turbo", "large-v3"]
                if model_size in size_hierarchy:
                    model_idx = size_hierarchy.index(model_size)
                    optimal_idx = size_hierarchy.index(model_option) if model_option in size_hierarchy else len(
                        size_hierarchy)
                    return model_size if model_idx <= optimal_idx else model_option
                return model_option
        # Really low gpu memory fallback
        return "small"

    def initialize_model(self, model_size: str, use_gpu: bool, speedup: bool) -> str:
        """Function to choose and load a preferred model."""
        # Check if model is already loaded when switching models
        if self.model is not None:
            self.print_info("Switching models...", message_type="info")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

        # Load transcription model
        model_str = self._select_model(model_size, use_gpu)

        try:
            device = self.device if use_gpu else "cpu"
            if use_gpu and torch.cuda.is_available():
                compute_type = "int8_float16" if speedup else "float16"
            else:
                compute_type = "auto"

            self.model = WhisperModel(model_str, device=device, compute_type=compute_type)
            # Return status message
            device_str = "GPU" if device == "cuda" else "CPU"
            return f"Using {model_str.upper()} model on {device_str}"

        except RuntimeError as e:
            self.print_info(f"Model initialization error: {str(e)}", message_type="error")
            return f"Model initialization error: {str(e)}"

    def _save_transcript(self, transcript: List[str], filename: str) -> str:
        """Save the transcript to a file."""
        output_path = Path("./output") / f"{filename}.txt"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(transcript))
            self.print_info(f"Transcript saved to {output_path}", message_type="info")
            return str(output_path)

        except Exception as e:
            error_msg = f"Error saving transcript for {filename}: {e}"
            self.print_info(error_msg, message_type="error")
            raise RuntimeError(error_msg)

    def _transcribe(self, audio_path: str, timestamps: bool, progress=gr.Progress(track_tqdm=True)) -> List[str]:
        """Transcribe the audio and return the transcript."""
        if not self.model:
            self.print_info("Model not initialized", message_type="warning")
            raise RuntimeError("Model not initialized.")

        transcript, info = self.model.transcribe(audio_path, vad_filter=True)
        self.print_info(f"Finished transcribing file. Decoding...", message_type="info")
        output_lines = []
        processed_duration = 0.0

        try:
            with tqdm(total=info.duration, unit=" s") as pbar:
                for segment in transcript:
                    # Combat potential hallucinations
                    if segment.end > info.duration:
                        break

                    if timestamps:
                        timestamp_str = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
                        output_lines.append(timestamp_str + segment.text.strip())
                    else:
                        output_lines.append(segment.text.strip())

                    # Update progress bar
                    pbar.update(segment.end - processed_duration)
                    processed_duration = segment.end
                    progress(processed_duration / info.duration,
                             desc=f"Processed {processed_duration:.1f}s / {info.duration:.1f}s")

                # Handle silence at the end
                if processed_duration < info.duration:
                    pbar.update(info.duration - processed_duration)
                    progress(1.0, desc=f"Completed: {info.duration:.1f}s / {info.duration:.1f}s")

        except Exception as e:
            error_msg = f"Error during decoding: {e}"
            self.print_info(error_msg, message_type="warning")
            raise RuntimeError(error_msg)

        return output_lines

    def _extract_audio(self, video_path: str, output_path: str) -> None:
        """Extract audio from a video file and save it as a wav file."""
        try:
            with VideoFileClip(video_path) as clip:
                clip.audio.write_audiofile(output_path, logger=None)
        except Exception as e:
            error_msg = f"Error extracting audio from video: {e}"
            self.print_info(error_msg, message_type="error")
            raise RuntimeError(error_msg)

    def _convert_audio_to_wav(self, audio_path: str, output_path: str) -> None:
        """Convert audio file to wav format if necessary."""
        try:
            with AudioFileClip(audio_path) as audio:
                audio.write_audiofile(output_path, logger=None)
        except Exception as e:
            error_msg = f"Error converting audio to wav: {e}"
            self.print_info(error_msg, message_type="error")
            raise RuntimeError(error_msg)

    def process_files(self, files: List[gr.File], timestamps: bool) -> str:
        if not files:
            self.print_info("No files provided for transcription.", message_type="warning")
            return "No files provided for transcription."

        start_time = datetime.now()

        try:
            for i, file in enumerate(files):
                file_path = Path(file.name)
                filename = file_path.stem

                # Filetype check
                if file_path.suffix.lower() not in self.video_formats and file_path.suffix.lower() not in self.audio_formats:
                    self.print_info(f"Unsupported file format: {file_path.suffix}. Skipping file...",
                                    message_type="warning")
                    continue

                # Audio file
                elif file_path.suffix.lower() in self.audio_formats:
                    self.print_info(f"Audio file detected: {file_path.name}", message_type="info")

                    # Convert to wav if not already
                    if file_path.suffix.lower() != ".wav":
                        audio_path = Path(f"./audiodata/{filename}.wav")
                        self._convert_audio_to_wav(file_path, audio_path)
                        process_path = str(audio_path)
                    else:
                        process_path = str(file_path)
                # Video file
                else:
                    self.print_info(f"Video file detected: {file_path.name}", message_type="info")

                    # Extract audio
                    process_path = Path("./audiodata") / f"{filename}.wav"
                    self._extract_audio(str(file_path), str(process_path))

                transcript = self._transcribe(process_path, timestamps)
                self._save_transcript(transcript, filename)

        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            self.print_info(error_msg, message_type="error")
            return error_msg

        total_time = (datetime.now() - start_time).total_seconds()

        return (f"Processed all file(s) in {total_time:.1f}s\n"
                f"Transcripts saved in ./output/ folder")

    def create_interface(self):
        """Create the Gradio interface for Audioscribe."""
        # speedy_transcription
        css = """
        .gradio-container {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .tabs {margin-top: 20px;}
        """

        model_info = {
            "small": f"Small model, fast but less accurate (~530MB)",
            "medium": "Medium model, more balanced speed/accuracy (~1525MB)",
            "large-v3-turbo": "Higher accuracy, optimized speed (~1620MB)",
            "large-v3": "High accuracy, slower (~3030MB)"
        }
        model_choices = [f"{k} - {v}" for k, v in model_info.items()]

        with gr.Blocks(css=css, title="Audioscribe") as interface:
            gr.Markdown("# Audioscribe")
            gr.Markdown("An audio and video transcription tool using Whisper AI.")

            with gr.Tabs():
                with gr.TabItem("Setup"):
                    gr.Markdown("## Model configuration")

                    with gr.Row(equal_height=True):
                        with gr.Column():
                            model_dropdown = gr.Dropdown(
                                label="Model Size",
                                choices=model_choices,
                                value=model_choices[-2],  # Default to large-v3-turbo
                                info="Larger models are more accurate, but slower"
                            )
                            with gr.Row():
                                gpu_checkbox = gr.Checkbox(
                                    label="Use GPU",
                                    value=True if torch.cuda.is_available() else False,
                                    info=f"GPU available: {torch.cuda.is_available()}"
                                )

                                speedup_checkbox = gr.Checkbox(
                                    label="Speed up transcription",
                                    value=False,
                                    info="Trade transcription quality for speed"
                                )

                            init_button = gr.Button("Initialize Model", variant="primary")
                            status_output = gr.Textbox(label="Status", interactive=False)

                        with gr.Column():
                            gr.Markdown("""
                            ## Getting Started
                            1. **Select Model**: 
                                - Choose a model size based on your needs (accuracy vs speed).
                                - Enable GPU support if available for faster processing.
                                - Click "Initialize Model" to load the selected model.
                            2. **Transcribe Files**:
                                - Switch to the "Transcribe" tab.
                                - Upload audio and/or video files; Audioscribe can handle both.
                                - Optionally, enable timestamps to include segment timings in the transcript.
                                - Start the transcription by clicking the "Transcribe" button.
                            3. **View Results**:
                                - Your transcripts will be saved automatically when complete.
                                
                            ### Tips for Best Results
                            - Use high-quality audio
                            - Minimize background noise
                            - Larger models provide better accuracy but are slower
                            """)

                            gr.Markdown(f"""
                            ### System Information
                            - **GPU Available**: {"Yes" if torch.cuda.is_available() else "No"}
                            - **Device Memory**: {self.memory / 2 ** 30 if self.memory > 0 else 0:.1f} GB
                            - **Output Directory**: ./output/
                            - **Audio Directory**: ./audiodata/
                            """)

                    def initialize_model_wrapper(model_choice: str, use_gpu: bool, speedup: bool) -> str:
                        model_str = model_choice.split(" - ")[0]
                        return self.initialize_model(model_str, use_gpu, speedup)

                    init_button.click(
                        fn=initialize_model_wrapper,
                        inputs=[model_dropdown, gpu_checkbox, speedup_checkbox],
                        outputs=status_output
                    )

                with gr.TabItem("Transcribe"):
                    gr.Markdown(f"### Upload files for transcription\n"
                                f"Supported formats: {', '.join(self.audio_formats | self.video_formats)}")

                    files = gr.Files(
                        file_types=["audio", "video"],
                        type="filepath",
                        label="Upload Audio/Video Files"
                    )

                    transcribe_timestamps = gr.Checkbox(
                        label="Include Timestamps",
                        value=False,
                        info="Add timestamps to each transcript segment"
                    )

                    transcribe_button = gr.Button("Transcribe", variant="primary")
                    transcribe_output = gr.Textbox(label="Results", interactive=False, lines=5)

                    transcribe_button.click(
                        fn=self.process_files,
                        inputs=[files, transcribe_timestamps],
                        outputs=transcribe_output
                    )

        return interface

    def launch_app(self, **kwargs):
        """Launch the Gradio app."""
        interface = self.create_interface()
        interface.launch(inbrowser=True, show_error=True, **kwargs)


def main():
    """Main function to run Audioscribe."""
    try:
        app = Audioscribe()
        app.launch_app()
    except Exception as e:
        print(f"Application error: {e}")
        raise


if __name__ == '__main__':
    main()
