from typing import Dict, Any, Optional, List, Tuple
from pydub.silence import split_on_silence
from pydub import AudioSegment
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import subprocess
import scrapetube
import datetime
import argparse
import re
import os


def extract_video_info(video_data):
    """
    Extract the video URL and cleaned title from the video data.

    Parameters:
    - video_data: The raw video data from YouTube.

    Returns:
    - dict: Dictionary with "url" and "title" keys.
    """
    base_video_url = "https://www.youtube.com/watch?v="
    video_id = video_data["videoId"]
    video_title = video_data["title"]["runs"][0]["text"]
    return {"url": f"{base_video_url}{video_id}", "title": video_title}


def get_video_urls(
    channel_or_playlist_id: str, pattern: str = None
) -> List[Dict[str, str]]:
    """
    Get video URLs and cleaned video titles either from a channel with a specified pattern or from a playlist.
    """
    if "youtube.com/playlist" in channel_or_playlist_id:
        playlist_id = channel_or_playlist_id.split("?list=")[-1].split("&")[0]
        videos = scrapetube.get_playlist(playlist_id)
    else:
        videos = scrapetube.get_channel(channel_or_playlist_id)

    if pattern:
        videos = [
            video
            for video in videos
            if re.search(pattern, video["title"]["runs"][0]["text"])
        ]

    return [extract_video_info(video) for video in videos]


def get_video_urls_from_channel(channel_id: str, pattern: str = None) -> List[str]:
    """
    Get video URLs from a channel with a specified pattern.
    """
    return [video["url"] for video in get_video_urls(channel_id, pattern)]


def get_video_urls_from_playlist(playlist_id: str) -> List[str]:
    """
    Get video URLs from a playlist.
    """
    return [video["url"] for video in get_video_urls(playlist_id)]


class Transcriber:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize a Transcriber object.

        Parameters:
        - api_key (str) : OpenAI API key.
        """
        if api_key is not None:
            self.client = OpenAI(api_key=api_key)

        self.supported_formats = [
            "wav",
            "mp3",
            "ogg",
            "flv",
            "m4a",
            "mp4",
            "wma",
            "aac",
        ]

    def convert_mp4_to_m4a(self, input_file: str, output_file: str) -> None:
        """
        Converts an MP4 file to M4A.

        Parameters:
        - input_file (str): Path to the input MP4 file.
        - output_file (str): Path to save the converted M4A file.
        """
        try:
            command = [
                "ffmpeg",
                "-i",
                input_file,
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                output_file,
            ]
            subprocess.run(command, check=True)
        except Exception as e:
            print(f"Error converting MP4 to M4A: {e}")
            return None

    def preprocess_audio(
        self,
        input_file: str,
        output_file: str,
        audio_format: str = "m4a",
        channels: int = 1,
        frame_rate: int = 16000,
        min_silence_len: int = 500,
        silence_thresh: int = -40,
        keep_silence: int = 200,
    ) -> None:
        """
        Preprocesses audio by splitting on silence and exports to WAV.

        Parameters:
        - input_file (str): Path to the input audio file.
        - output_file (str): Path to save the processed audio file.
        - audio_format (str): Format of the input audio file.
        - channels (int): Desired number of channels for the processed audio.
        - frame_rate (int): Desired frame rate for the processed audio.
        - min_silence_len (int): Minimum length of silence to split on.
        - silence_thresh (int): Threshold value for silence detection.
        - keep_silence (int): Amount of silence to retain around detected non-silent chunks.
        """
        if audio_format not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {audio_format}. Supported formats are: {', '.join(self.supported_formats)}"
            )
        try:
            audio = (
                AudioSegment.from_file(input_file, format=audio_format)
                .set_channels(channels)
                .set_frame_rate(frame_rate)
            )
            audio_segments = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence,
            )
            processed_audio = AudioSegment.empty()
            for segment in audio_segments:
                processed_audio += segment
            processed_audio.export(output_file, format="wav")
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            return None

    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribes audio using the Whisper model.

        Parameters:
        - audio_file_path (str): Path to the audio file to transcribe.

        Returns:
        - Dict[str, Any]: Transcription results.
        """
        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=Path(audio_file_path), response_format="text"
            )
            return transcript
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {}

    def _run_yt_dlp_with_progress(self, cmd: list, description: str) -> int:
        """
        Helper function to run yt-dlp command with tqdm progress bar.

        Parameters:
        - cmd (list): Command and its arguments for yt-dlp.
        - description (str): Description for the tqdm progress bar.

        Returns:
        - int: Return code of the subprocess.
        """
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        ) as process, tqdm(total=100, unit="%", desc=description, leave=False) as pbar:
            for line in process.stdout:
                print(line, end="")
                if "download" in line.lower():
                    percentage = re.findall(r"\d+\.\d+%", line)
                    if percentage:
                        progress = float(percentage[0].replace("%", ""))
                        pbar.n = progress
                        pbar.refresh()
        return process.returncode

    def download_youtube_audio(
        self, youtube_url: str, output_path: str, custom_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Downloads audio from a YouTube video using yt-dlp with tqdm progress bar.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        format_selector = (custom_format if custom_format else "bestaudio[ext=m4a]",)

        cmd = [
            "yt-dlp",
            "-f",
            format_selector,
            "-o",
            os.path.join(output_path, "%(title)s.%(ext)s"),
            "--newline",
            youtube_url,
        ]

        if self._run_yt_dlp_with_progress(cmd, "Downloading Audio") == 0:
            file_path = os.path.join(
                output_path, "downloaded_audio.m4a"
            )  # Modify as needed
            print("Downloaded audio path:", file_path)
            return file_path
        return None

    def download_youtube_video(
        self, youtube_url: str, output_path: str, custom_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Downloads video from a YouTube video using yt-dlp and converts it to a QuickTime compatible format.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        format_selector = custom_format if custom_format else "best"
        cmd = [
            "yt-dlp",
            "-f",
            format_selector,
            "-o",
            os.path.join(output_path, "%(title)s.%(ext)s"),
            youtube_url,
        ]

        if self._run_yt_dlp_with_progress(cmd, "Downloading Video") == 0:
            file_path = os.path.join(
                output_path, f"{os.path.basename(youtube_url)}.mp4"
            )
            print("Downloaded video path:", file_path)

            # Convert the video to a QuickTime compatible format using ffmpeg (if needed)
            return file_path
        return None

    def download_youtube_media(
        self,
        youtube_url: str,
        output_path: str,
        media_type: str = "audio",
        custom_format: Optional[str] = None,
    ) -> Optional[str]:
        """
        Downloads audio or video from a YouTube video using yt-dlp.

        Parameters:
        - youtube_url (str): URL of the YouTube video.
        - output_path (str): Path to save the downloaded audio or video file.
        - media_type (str): Type of media to download, can be 'audio' or 'video'.
        - custom_format (str, optional): Custom format to download the media in.

        Returns:
        - Optional[str]: Path to the downloaded audio or video file or None if unsuccessful.
        """
        if media_type == "audio":
            return self.download_youtube_audio(
                youtube_url, output_path, custom_format=custom_format
            )
        elif media_type == "video":
            return self.download_youtube_video(
                youtube_url, output_path, custom_format=custom_format
            )
        else:
            raise ValueError(
                f"Unsupported media type: {media_type}. Supported types are: audio, video"
            )

    def download_multiple_youtube_media(
        self,
        youtube_urls: List[str],
        output_path: str,
        media_type: str = "audio",
        custom_format: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Downloads audio or video from a list of YouTube videos using download_youtube_media.

        Parameters:
        - youtube_urls (List[str]): List of URLs of the YouTube videos.
        - output_path (str): Path to save the downloaded audio or video files.
        - media_type (str): Type of media to download, can be 'audio' or 'video'.

        Returns:
        - List[Tuple[str, Optional[str]]]: A list of tuples where the first element is the YouTube URL and
        the second element is the name of the downloaded file or None if unsuccessful.
        """
        downloaded_files = []

        for url in youtube_urls:
            try:
                filename = self.download_youtube_media(
                    url, output_path, media_type, custom_format
                )
                downloaded_files.append((url, filename))
            except ValueError as ve:
                print(f"Failed to download {url}: {ve}")
                downloaded_files.append((url, None))
            except Exception as e:
                print(f"An error occurred while downloading {url}: {e}")
                downloaded_files.append((url, None))

        return downloaded_files

    def download_videos_from_source(
        self,
        source_id: str,
        output_path: str,
        media_type: str = "audio",
        pattern: str = None,
        custom_format: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Download media from YouTube videos given a source, which can be a channel or playlist ID or URL.

        Parameters:
        - source_id (str): YouTube channel ID, playlist ID, or URL.
        - output_path (str): Path to save the downloaded media.
        - media_type (str): Type of media to download, 'audio' or 'video'.
        - pattern (str, optional): Pattern to match video titles (for channels).

        Returns:
        - List[Tuple[str, Optional[str]]]: List of tuples containing the video URL and the filename of the downloaded media.
        """
        if "youtube.com/playlist" in source_id:
            video_urls = get_video_urls_from_playlist(source_id)
        else:
            video_urls = get_video_urls_from_channel(source_id, pattern)

        return self.download_multiple_youtube_media(
            video_urls, output_path, media_type, custom_format
        )

    def transcribe(
        self,
        input_file: str,
        output_file: str,
        audio_format: str = "m4a",
        save_to_csv: bool = False,
        use_preprocessing: bool = False,
        convert_from_mp4: bool = False,
    ) -> pd.DataFrame:
        """
        Manages the transcription process and returns results as a DataFrame, with each sentence as a new row.
        """
        if convert_from_mp4:
            m4a_file = os.path.splitext(input_file)[0] + ".m4a"
            self.convert_mp4_to_m4a(input_file, m4a_file)
            input_file = m4a_file

        t1 = datetime.datetime.now()
        audio_file_to_transcribe = input_file

        if use_preprocessing:
            self.preprocess_audio(input_file, output_file, audio_format=audio_format)
            audio_file_to_transcribe = output_file

        transcription_result = self.transcribe_audio(audio_file_to_transcribe)
        t2 = datetime.datetime.now()

        # Check if the result is a string or a dict with a 'transcript' key
        if isinstance(transcription_result, str):
            transcript = transcription_result
        elif "transcript" in transcription_result and isinstance(
            transcription_result["transcript"], str
        ):
            transcript = transcription_result["transcript"]
        else:
            print("Unexpected format in transcription result:", transcription_result)
            return pd.DataFrame()  # Return an empty DataFrame or handle as appropriate

        sentences = transcript.split(". ")
        data = {
            "Sentence": sentences,
            "Start_Time": [t1] * len(sentences),
            "End_Time": [t2] * len(sentences),
            "Duration": [(t2 - t1).total_seconds()] * len(sentences),
        }

        output_df = pd.DataFrame(data)

        if save_to_csv:
            output_df.to_csv(output_file, index=False)

        return output_df

    def extract_audio_from_video(self, video_file: str, output_file: str) -> None:
        """
        Extracts audio from a video file using ffmpeg.

        Parameters:
        - video_file (str): Path to the video file to extract audio from.
        - output_file (str): Path to save the extracted audio file.
        """
        try:
            command = [
                "ffmpeg",
                "-i",
                video_file,
                "-ab",
                "160k",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-vn",
                output_file,
            ]
            subprocess.run(command, check=True)
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None

    def transcribe_video(
        self,
        video_file: str,
        output_file: str,
        save_to_csv: bool = False,
        use_preprocessing: bool = False,
    ) -> pd.DataFrame:
        """
        Manages the transcription of video by first extracting audio and then transcribing it.

        Parameters:
        - video_file (str): Path to the video file to transcribe.
        - output_file (str): Path to save the transcriptions.
        - save_to_csv (bool): Whether to save the results to a CSV file.
        - use_preprocessing (bool): Whether to preprocess the audio before transcription.

        Returns:
        - pd.DataFrame: Transcription results.
        """
        audio_file_path = os.path.splitext(video_file)[0] + ".wav"
        self.extract_audio_from_video(video_file, audio_file_path)

        return self.transcribe(
            audio_file_path,
            output_file,
            audio_format="wav",
            save_to_csv=save_to_csv,
            use_preprocessing=use_preprocessing,
        )

    def transcribe_batch(
        self,
        input_files: List[str],
        output_folder: str,
        audio_format: str = "m4a",
        save_to_csv: bool = False,
    ) -> pd.DataFrame:
        """
        Batch transcribes a list of audio files.

        Parameters:
        - input_files (List[str]): List of paths to audio files to transcribe.
        - output_folder (str): Path to save the transcriptions.
        - audio_format (str): Format of the input audio files.
        - save_to_csv (bool): Whether to save the results to a CSV file.

        Returns:
        - pd.DataFrame: Batch transcription results.
        """
        dfs = []

        for i, input_file in enumerate(input_files):
            base_name = os.path.basename(input_file)
            file_name_without_extension = os.path.splitext(base_name)[0]
            output_file = os.path.join(
                output_folder, f"{file_name_without_extension}_processed.wav"
            )
            print(f"Processing file {i+1}/{len(input_files)}: {input_file}")
            df = self.transcribe(input_file, output_file, audio_format=audio_format)
            dfs.append(df)

        result_df = pd.concat(dfs, ignore_index=True)
        if save_to_csv:
            result_csv_path = os.path.join(output_folder, "batch_transcriptions.csv")
            result_df.to_csv(result_csv_path, index=False)
            print(f"Saved batch transcriptions to: {result_csv_path}")
        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Download and optionally transcribe media from YouTube."
    )
    parser.add_argument(
        "youtube_url", type=str, help="The YouTube URL to download from."
    )
    parser.add_argument(
        "output_path", type=str, help="The directory to save the downloaded media."
    )
    parser.add_argument(
        "media_type", type=str, help="The type of media to download, audio or video."
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Whether to transcribe the downloaded media.",
    )
    args = parser.parse_args()

    # Initialize the Transcriber
    transcriber = Transcriber(api_key=args.api_key)

    # Run the download_youtube_media method
    file_path = None
    try:
        transcriber.download_youtube_media(
            args.youtube_url, args.output_path, media_type=args.media_type
        )
        if file_path:
            print("Downloaded file path:", file_path)
        else:
            print("Download failed or no file was downloaded.")
    except ValueError as ve:
        print(f"Failed to download {args.youtube_url}: {ve}")

    # Run the transcribe_video method only if file_path is not None
    if args.transcribe and file_path:
        try:
            output_file = os.path.join(args.output_path, "transcription.csv")
            transcriber.transcribe_video(file_path, output_file, save_to_csv=True)
            print(f"Transcription saved to: {output_file}")
        except Exception as e:
            print(f"An error occurred during transcription: {e}")


if __name__ == "__main__":
    main()
