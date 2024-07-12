import os
import json
import uuid
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio
from torchvision.models import resnet18
from b2aiprep.process import Audio, specgram, plot_spectrogram, plot_waveform
import IPython.display as Ipd
#from torchsummary import summary

import numpy as np
from pydub import AudioSegment
import librosa


def determine_is_stridor_and_phonatory(name, folder_path, patient_status_df):
    """
    Determine if the path should be marked as stridor and phonatory based on the folder structure and excel sheet.
    """
    for part in folder_path.split(os.sep):
        if part.upper() in ["CONTROL", "CONTROLS"]:
            return "No-Stridor", "Non-Phonatory", "No-Stridor"  # stridor_status, phonatory_status, status

    # Extract patient code from the name
    if name.startswith("Patient "):
        patient_code = int(name.split(" ")[1])
        if patient_code in patient_status_df['Code'].values:
            status = patient_status_df[patient_status_df['Code'] == patient_code]['Status'].values[0]
            if status == "No-Stridor":
                return "No-Stridor", "No-Stridor", "No-Stridor"  # stridor_status, phonatory_status, status
            elif status == "Phonatory":
                return "Stridor", "Phonatory", "Phonatory"  # stridor_status, phonatory_status, status
            elif status == "Non-Phonatory":
                return "Stridor", "Non-Phonatory", "Non-Phonatory"  # stridor_status, phonatory_status, status
    
    return "No-Stridor", "No-Stridor", "No-Stridor"  # Default to non-stridor if not found

def get_audio_duration(file_path):
    """
    Get the duration of the audio file in seconds.
    """
    audio = AudioSegment.from_wav(file_path)
    return len(audio) / 1000  # Convert milliseconds to seconds

def load_audio(file_path):
    """
    Load the audio file.
    """
    y, sr = librosa.load(file_path)
    return y, sr

def compute_short_time_energy(y, frame_size=1024, hop_length=512):
    """
    Compute the short-time energy of the audio signal.
    """
    energy = torch.tensor([
        torch.sum(y[i:i+frame_size] ** 2)
        for i in range(0, len(y), hop_length)
    ])
    return energy

def get_most_active_segment(signal, sr, required_samples, hop_length=512):
    # Compute short-time energy
    frame_size = hop_length
    energy = compute_short_time_energy(signal, frame_size, hop_length)
    
    # Ensure required_samples is an integer
    required_samples = int(required_samples)
    
    # Find the most active segment
    active_start_frame = torch.argmax(torch.nn.functional.conv1d(
        energy.view(1, 1, -1), 
        torch.ones(1, 1, required_samples // hop_length)
    ))
    active_start_sample = active_start_frame * hop_length
    active_end_sample = active_start_sample + required_samples

    return signal[active_start_sample:active_end_sample]

def generate_recording_objects(root_path, patient_status_df, label_column):
    recordings = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(root_path):
        # Check if there are wav files in the current directory
        wav_files = [file for file in files if file.endswith('.wav')]
        if wav_files:
            # Get the parent folder name
            parent_folder = os.path.basename(root)
            
            for wav_file in wav_files:
                if parent_folder in ["Patient 1", "Patient 5", "Patient 10"]:
                    # print(f"parent_folder: {parent_folder}")
                    continue
                
                # Generate a unique ID for each player-session-recording combination
                uid = str(uuid.uuid4())
                # Extract the recording label (base name without extension)
                recording_label = os.path.splitext(wav_file)[0]
                # Get the full path of the wav file
                file_path = os.path.join(root, wav_file)
                # Get the duration of the audio file
                duration = get_audio_duration(file_path)

               
                
                # # Load the audio file and compute its short-time energy
                # y, sr = load_audio(file_path)
                # slots = find_most_active_timeframe(y, sr, hop_length=512, total_duration=total_duration, slot_duration=audio_sample_duration)
                
                # Determine is_stridor and is_phonatory from the folder structure and Excel sheet
                stridor_status, phonatory_status, status = determine_is_stridor_and_phonatory(parent_folder, root, patient_status_df)
                
                if label_column == "phonatory_status" and phonatory_status == "No-Stridor":
                    # if we are checking for phonatory status and the patient belongs to No-Stridor batch, then it shouldn't be included in the phonatory check
                    continue
                
                # # Calculate the number of slots and their duration
                # num_slots = duration // 3
                # if num_slots == 0:
                #     continue  # Skip recordings that are too short
                # total_duration = num_slots * 3
                
                # Create the entry for the current recording
                entry = {
                    "uid": uid,
                    "recording_id": uid,
                    "name": parent_folder,
                    "recording": file_path,  # Include the full path
                    "recording_label": recording_label,
                    "stridor_status": stridor_status,
                    "phonatory_status": phonatory_status,
                    "status": status,
                    "duration": duration,
                    # "recording_slots": slots
                }
                recordings.append(entry)
                
    return recordings

def save_to_csv(recordings, output_csv):
    # Flatten the recording slots for CSV
    flattened_data = []
    for record in recordings:
        base_data = {key: record[key] for key in record if key != 'recording_slots'}
        for slot in record['recording_slots']:
            slot_data = base_data.copy()
            slot_data.update(slot)
            flattened_data.append(slot_data)
    
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_csv, index=False)

def save_to_json(recordings, output_json):
    with open(output_json, 'w') as json_file:
        json.dump(recordings, json_file, indent=4)

def count_samples(recordings):
    no_stridor_count = 0
    phonatory_count = 0
    non_phonatory_count = 0

    for record in recordings:
        if record['status'] == "No-Stridor":
            no_stridor_count += 1
        elif record['status'] == "Phonatory":
            phonatory_count += 1
        elif record['status'] == "Non-Phonatory":
            non_phonatory_count += 1
    
    return no_stridor_count, phonatory_count, non_phonatory_count

def filter_recordings_by_label(data, label_prefix):
    """
    Filter recordings to only include those with recording_label starting with the specified prefix.
    Exclude specific labels for certain prefixes.
    Comparisons are case insensitive.
    """
    # Define exclusions for each prefix
    exclude_labels = {
        "FIMO": ["FIMOcricoid", "FIMOthyroid", "FIMOC", "FIMOT"],
        "RP": ["RPC", "RPT", "RPcricoid", "RPcriocid", "RPthyroid"]
    }

    # Convert prefix to lower case for case insensitive comparison
    label_prefix_lower = label_prefix.lower()
    
    # print(f"label_prefix_lower: {label_prefix_lower}")

    filtered_data = []
    for recording in data:
        # Get the recording label in lower case
        recording_label_lower = recording['recording_label'].lower()
        # print(f"recording_label_lower: {recording_label_lower}")
        
        # Check if the recording label starts with the given prefix
        if recording_label_lower.startswith(label_prefix_lower):
            # Check for exclusions
            if label_prefix in exclude_labels:
                if recording_label_lower in map(str.lower, exclude_labels[label_prefix]):
                    continue
            filtered_data.append(recording)

    return filtered_data