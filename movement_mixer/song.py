from typing import  List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import librosa
import json
import glob
import os


def pad_or_truncate_features(features, max_length):
    for key in features:
        if len(features[key][0]) < max_length:
            features[key] = [
                row + [0] * (max_length - len(row)) for row in features[key]
            ]
        elif len(features[key][0]) > max_length:
            features[key] = [row[:max_length] for row in features[key]]
    return features


def save_features_as_json(features, json_file):
    with open(json_file, "w") as json_output:
        json.dump(features, json_output, indent=4)


def load_features_from_json(json_file):
    with open(json_file, "r") as json_input:
        return json.load(json_input)

def get_files(directory: str, patterns: List[str]) -> List[str]:
    files = []
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        files.extend(glob.glob(full_pattern, recursive=True))
    return files

 

def convert_audio_to_images(
    input_dir,
    file_patterns= ["*.m4a"],
    output_dir="dj_spectrograms",
    figure_size=(6, 4),
    dpi=300,
    cmap="viridis",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = get_files(directory=input_dir, patterns=file_patterns)

    for file_path in file_paths:

        def get_spectrogram(file_path: str):
            y, sr = librosa.load(file_path, sr=44100)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            return librosa.power_to_db(S, ref=np.max), y, sr, S

        data, y, sr, S = get_spectrogram(file_path)

        # Adjust figure size, DPI, and colormap
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        librosa.display.specshow(data, cmap=cmap)

        # Generate the filename
        filename = os.path.join(
            output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".png"
        )

        # Save the image with specified DPI and options
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)

        # Clean up and release resources
        plt.close("all")
        plt.close(fig)
        del filename
        del fig
        del ax
        del data
        del y
        del sr
        del S


def calculate_features(file_path, n_fft=2048, hop_length=512, n_mels=128, fmax=8000):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=44100)

    # Calculate the mel-spectrogram, which is a common feature used in DJing for visualizing the frequency content of the music
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, hop_length=hop_length
    )

    # Beat tracking to find the tempo and the beats, which are essential for DJing to match the beats of two tracks
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Return a dictionary of features relevant for DJing house music
    return {
        "name": os.path.splitext(os.path.basename(file_path))[0],
        "tempo": tempo,
        "beats": beats.tolist(),
        "mel_spectrogram": S.tolist(),
    }


def process_audio_files(file_paths, output_dir="dj_features"):
    os.makedirs(output_dir, exist_ok=True)
    all_features = []
    max_length = 0

    for file_path in file_paths:
        features = calculate_features(file_path)
        all_features.append(features)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        json_file = os.path.join(output_dir, f"{file_name}_features.json")
        save_features_as_json(features, json_file)

    max_length = max(len(all_features[0]["chromagram"][0]), max_length)

    for json_file in os.listdir(output_dir):
        features = load_features_from_json(os.path.join(output_dir, json_file))
        features = pad_or_truncate_features(features, max_length)
        save_features_as_json(features, os.path.join(output_dir, json_file))

    combined_features = [
        load_features_from_json(os.path.join(output_dir, json_file))
        for json_file in os.listdir(output_dir)
    ]

    combined_json_filename = os.path.join(output_dir, "combined_audio_features.json")
    save_features_as_json(combined_features, combined_json_filename)

    return combined_features


def convert_audio_to_wav(file_paths, output_dir="dj_wav"):
    os.makedirs(output_dir, exist_ok=True)

    def process_audio(file_path):
        try:
            y, sr = librosa.load(file_path, sr=44100)

            # Generate the WAV filename
            filename = os.path.join(
                output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".wav"
            )

            # Save the audio as a WAV file
            librosa.write(filename, y, sr)

            # Print a message indicating the conversion
            print(f"Converted {file_path} to {filename}")

            # Extract some basic audio features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            # Store the audio features in a DataFrame
            features = {
                "file_path": file_path,
                "tempo": tempo,
                "beats": beats.tolist(),
                "chromagram": chromagram.tolist(),
                "tonnetz": tonnetz.tolist(),
            }
            features_df = pd.DataFrame(features)
            features_csv_filename = os.path.join(output_dir, "audio_features.csv")
            features_df.to_csv(features_csv_filename, index=False)

            # Play the converted audio
            # ipd.display(ipd.Audio(filename))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    for file_path in file_paths:
        process_audio(file_path)

class Song:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.audio = None
        self.sample_rate = 44100
        self.beats = None
        self.downbeats = None

        if filepath is not None:
            self.song_name, self.song_format = self.get_song_name_and_format()
            self.load_song_audio()
            self.load_beats()

    def plot_downbeats(self, start_dbeat, end_dbeat, plot_name="", color="red"):
        plt.rcParams["figure.figsize"] = (20, 9)
        dbeats = self.get_downbeats()
        start_idx, end_idx = int(dbeats[start_dbeat]), int(dbeats[end_dbeat])
        selected_dbeats = dbeats[start_dbeat : end_dbeat + 1] - start_idx
        plt.plot(self.audio[start_idx:end_idx])
        for dbeat in selected_dbeats:
            plt.axvline(dbeat, color=color)
        plt.title(plot_name)
        plotname = "".join(plot_name.split(" "))
        plt.savefig(f"{plotname}.png")

    def load_song_audio(self):
        self.audio, self.sample_rate = librosa.load(self.filepath, sr=self.sample_rate)

    def get_song_name_and_format(self):
        return os.path.basename(self.filepath).split(".")

    def annotate_beats(self, output_filepath):
        tempo, beats = librosa.beat.beat_track(y=self.audio, sr=self.sample_rate)
        np.savetxt(output_filepath, beats, newline="\n")
        return beats

    def get_downbeats(self):
        if self.downbeats is not None:
            return self.downbeats

        # Implement a basic downbeat detection algorithm or use beats directly
        # This is a placeholder and should be replaced with a more robust method
        dbeats = self.beats[
            ::4
        ]  # Assuming 4/4 time, take every fourth beat as a downbeat
        dbeats_time_to_audio_index = np.array(dbeats, dtype=float) * self.sample_rate
        self.downbeats = np.array(dbeats_time_to_audio_index, dtype=int)
        return self.downbeats




def main():
    parser = argparse.ArgumentParser(
        description="Convert Audio to Image."
    )
    parser.add_argument(
        "input_dir", type=str, help="The YouTube URL to download from."
    )
    
    parser.add_argument(
        "output_dir", help="The directory to save the downloaded media."
    )

    args = parser.parse_args()

    convert_audio_to_images(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
