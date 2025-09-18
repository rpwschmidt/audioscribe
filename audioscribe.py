import gc
import torch
import whisperx
import gradio as gr
from pathlib import Path
from typing import List, Set
from datetime import datetime
from moviepy import VideoFileClip, AudioFileClip


class Audioscribe:
    """Main Audioscribe class to handle the transcription of audio and video files."""
    def __init__(self):
        self.model = None
        self.model_str = None
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
    def _format_time(seconds: float) -> str:
        """Convert seconds to hh:mm:ss format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = round(seconds % 60, 2)
        return f"{hours:02}:{minutes:02}:{secs:04}"

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
                # Fix reproducibility issue 
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
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

    def _unload_model(self, reason: str = "") -> None:
        self.print_info(f"Switching models{reason}...", message_type="info")
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()

    def initialize_model(self, model_size: str, use_gpu: bool, speedup: bool) -> str:
        """Function to choose and load a preferred model."""
        # Check if model is already loaded when switching models
        if self.model is not None:
            self._unload_model()

        # Load transcription model
        self.model_str = self._select_model(model_size, use_gpu)

        try:
            device = self.device if use_gpu else "cpu"
            if use_gpu and torch.cuda.is_available():
                compute_type = "int8_float16" if speedup else "float16"
            else:
                compute_type = "auto"

            self.model = whisperx.load_model(self.model_str, device=device, compute_type=compute_type)
            # Return status message
            device_str = "GPU" if device == "cuda" else "CPU"
            return f"Using {self.model_str.upper()} model on {device_str}"

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

    def _align(self, result, audio_path, device):
        self._unload_model(reason=", identifying speakers. Aligning text")
        self.model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        alignment_result = whisperx.align(result["segments"], self.model, metadata, audio_path, device, return_char_alignments=False)
        return alignment_result

    def _sync_output(self, result, audio_path, device, num_speakers):
        with open("./hf_token.txt") as file:
            hf_token = file.read().strip()

        self._unload_model(reason=", getting speaker segments")
        self.model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = self.model(audio_path, num_speakers=num_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        return result

    def _swap_back(self, device, compute_type):
        self._unload_model(reason=", switching back to transcription model")
        self.model = whisperx.load_model(self.model_str, device=device, compute_type=compute_type)
        self.print_info(f"Loaded transcription model.", message_type="info")

    def _speaker_diarization(self, result: dict, audio_path: str, num_speakers: int) -> dict:
        """Swap to a speaker diarization model if available."""
        device, compute_type = self.model.model.model.device, self.model.model.model.compute_type
        try:
            if Path('./hf_token.txt').is_file():
                # Align output
                alignment_result = self._align(result, audio_path, device)
                # Get speaker segments and assign to words
                result = self._sync_output(alignment_result, audio_path, device, num_speakers)
                self._swap_back(device, compute_type)
                return result
            else:
                self.print_info("HF token file not found. Speaker diarization disabled.", message_type="warning")

        except Exception as e:
            self.print_info(f"Error loading diarization model: {e}", message_type="error")

    def _format_output(self, start: str, end: str, speaker: str, text: str, timestamp: str) -> str:
        if timestamp:
            ts = f"[{self._format_time(start)}] - [{self._format_time(end)}]"
            return f"{ts} {speaker}: {text}" if speaker else f"{ts} {text}"
        return f"{speaker}: {text}" if speaker else text

    def _transcribe(self, audio_path: str, timestamps: bool, translate: bool, diarization: bool, num_speakers: int, progress=gr.Progress()) -> List[str]:
        """Transcribe the audio and return the transcript."""
        if not self.model:
            self.print_info("Model not initialized", message_type="warning")
            raise RuntimeError("Model not initialized.")

        progress(0.0, desc="Starting transcription...")
        task = "translate" if translate else "transcribe"
        result = self.model.transcribe(audio_path, task=task)

        if diarization:
            progress(0.7, desc="Identifying speakers...")
            result = self._speaker_diarization(result, audio_path, num_speakers)
            current_speaker = None
            current_text = []
            block_start = None
            block_end = None

        output_lines = []
        duration = result['segments'][-1]['end']
        progress(0.9, desc="Writing to file...")

        try:
            if diarization:
                for segment in result["segments"]:
                    # Combat hallucinations
                    if segment["end"] > duration:
                        break

                    speaker = segment.get("speaker", "UNKNOWN")
                    text = segment["text"].strip()

                    # Start a new block
                    if speaker != current_speaker:
                        # Save previous block
                        if current_speaker is not None:
                            output_lines.append(self._format_output(block_start, block_end, current_speaker, " ".join(current_text), timestamps))

                        # Reset block
                        current_speaker = speaker
                        current_text = [text]
                        block_start = segment["start"]
                        block_end = segment["end"]
                    else:
                        # Extend the current block
                        current_text.append(text)
                        block_end = segment["end"]

                # Flush the last block
                if current_speaker is not None:
                    output_lines.append(self._format_output(block_start, block_end, current_speaker, " ".join(current_text), timestamps))

            else:
                for segment in result["segments"]:
                    if segment['end'] > duration:
                        break
                    output_lines.append(self._format_output(segment['start'], segment['end'], None, segment['text'].strip(), timestamps))

        except Exception as e:
            error_msg = f"Error during decoding: {e}"
            self.print_info(error_msg, message_type="warning")
            raise RuntimeError(error_msg)

        progress(1.0, desc="Transcription complete.")
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

    def process_files(self, files: List[gr.File], timestamps: bool, translate: bool, diarization: bool, num_speakers: int) -> str:
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

                transcript = self._transcribe(process_path, timestamps, translate, diarization, num_speakers)
                self._save_transcript(transcript, filename)

        except Exception as e:
            error_msg = f"Error processing files: {str(e)}\nPlease try again."
            self.print_info(error_msg, message_type="error")
            return error_msg

        total_time = (datetime.now() - start_time).total_seconds()

        return (f"Processed all file(s) in {total_time:.1f}s\n"
                f"Transcripts saved in ./output/ folder")

    def create_interface(self):
        """Create the Gradio interface for Audioscribe."""
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
                                info="Larger models are more accurate, but slower.\nTranslation does NOT work with the turbo model!"
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
                            - When translating with `large-v3`, more than 4GB of GPU memory is required for it to work reliably. If you have less GPU memory and it crashes, please try again.
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
                    with gr.Row():
                        transcribe_timestamps = gr.Checkbox(
                            label="Include Timestamps",
                            value=False,
                            info="Add timestamps to each transcript segment"
                        )

                        transcribe_translate = gr.Checkbox(
                            label="Translate to English",
                            value=False,
                            info="Translate the transcript to English (if not already).\nDOES NOT WORK WITH THE TURBO MODEL!"
                        )

                        diarization_checkbox = gr.Checkbox(
                            label="Identify Speakers",
                            value=False,
                            info="Identify different speakers in the audio"
                        )

                        num_speakers = gr.Number(
                            label="Number of Speakers",
                            value=2,
                            info="Set the expected number of speakers (if known)",
                            minimum=1,
                            maximum=10,
                            step=1,
                            precision=0
                        )

                    transcribe_button = gr.Button("Transcribe", variant="primary")
                    transcribe_output = gr.Textbox(label="Results", interactive=False, lines=5)

                    transcribe_button.click(
                        fn=self.process_files,
                        inputs=[files, transcribe_timestamps, transcribe_translate, diarization_checkbox, num_speakers],
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
