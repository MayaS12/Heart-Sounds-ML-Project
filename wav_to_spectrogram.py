#!/usr/bin/python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Define the directory containing the audio data
dataset_dir = r'C:\Users\Student\Desktop\MayaS\Heart Sounds\Data'

# Define the output directory for normalized audio files
output_dir = r'C:\Users\Student\Desktop\MayaS\Heart Sounds\final_data'

# Define the audio parameters
sr = 16000  # Sampling rate

# Define the mapping from directory names to labels
label_map = {
    "N_New": "N",
    "AS_New": "AS",
    "MR_New": "MR",
    "MS_New": "MS",
    "MVP_New": "MVP",
}

def wav_to_spectrogram(wav_file, output_image=None, n_fft=2048, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    
    # Compute the spectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    # Save the spectrogram as an image if output_image is provided
    if output_image:
        plt.savefig(output_image)
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()
    
    return S_db

def process_directory(input_directory, output_directory, label_name, n_fft=2048, hop_length=512):
    # Create output subdirectory if it doesn't exist
    output_subdir = os.path.join(output_directory, label_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(input_directory, '*.wav'))
    
    for wav_file in wav_files:
        base_name = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_image = os.path.join(output_subdir, f'{name_without_ext}.png')
        
        wav_to_spectrogram(wav_file, output_image, n_fft, hop_length)

# Iterate over each label directory and process .wav files
for label_dir, label_name in label_map.items():
    full_label_dir = os.path.join(dataset_dir, label_dir)
    if os.path.exists(full_label_dir):
        output_label_dir = os.path.join(output_dir, label_name)
        process_directory(full_label_dir, output_label_dir, label_name)
    else:
        print(f"Directory not found: {full_label_dir}")

print("Finished processing files.")
