import json
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa as sf


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


def convert_audio_to_images(
    file_paths,
    output_dir="dj_spectrograms",
    figure_size=(6, 4),
    dpi=300,
    cmap="viridis",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
            sf.write(filename, y, sr)

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
