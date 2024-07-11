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
            return 0, 0, 0  # is_stridor, is_phonatory, status

    # Extract patient code from the name
    if name.startswith("Patient "):
        patient_code = int(name.split(" ")[1])
        if patient_code in patient_status_df['Code'].values:
            status = patient_status_df[patient_status_df['Code'] == patient_code]['Status'].values[0]
            if status == "No-Stridor":
                return 0, 0, 0  # is_stridor, is_phonatory, status
            elif status == "Phonatory":
                return 1, 1, 2  # is_stridor, is_phonatory, status
            elif status == "Non-Phonatory":
                return 1, 0, 1  # is_stridor, is_phonatory, status
            
            
    #global print_once
    #if print_once:
        #print(" WARNING: Patents 1,5, 10 do not found but gavien 0 , 0 , 0 status")
        #print_once = False
    #print('patient not found', name, folder_path)
    
    return 0, 0, 0  # Default to non-stridor if not found

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
    energy = np.array([
        np.sum(np.square(y[i:i+frame_size]))
        for i in range(0, len(y), hop_length)
    ])
    return energy

def find_most_active_timeframe(y, sr, hop_length, total_duration, slot_duration):
    """
    Find the most active timeframe in the audio signal and break it into slots.
    """
    frame_duration = int(slot_duration * sr / hop_length)
    total_frames = int(total_duration * sr / hop_length)
    energy = compute_short_time_energy(y, frame_size=hop_length, hop_length=hop_length)
    
    # Find the most active timeframe
    active_start_frame = np.argmax(np.convolve(energy, np.ones(total_frames), mode='valid'))
    active_start_time = (active_start_frame * hop_length) / sr
    active_end_time = active_start_time + total_duration

    # Break the active timeframe into slots
    slots = []
    for i in range(total_frames // frame_duration):
        slot_start_time = active_start_time + i * slot_duration
        slot_end_time = slot_start_time + slot_duration
        slots.append({"start_time": slot_start_time, "end_time": slot_end_time})
    
    return slots

def generate_recording_objects(root_path, patient_status_df, filtered_patiens):
    recordings = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(root_path):
        # Check if there are wav files in the current directory
        wav_files = [file for file in files if file.endswith('.wav')]
        if wav_files:
            # Get the parent folder name
            parent_folder = os.path.basename(root)
            
            for wav_file in wav_files:
                if parent_folder in filtered_patiens:
                    #print(f"parent_folder: {parent_folder}")
                    continue
                # Generate a unique ID for each player-session-recording combination
                uid = str(uuid.uuid4())
                # Extract the recording label (base name without extension)
                recording_label = os.path.splitext(wav_file)[0]
                # Get the full path of the wav file
                file_path = os.path.join(root, wav_file)
                # Get the duration of the audio file
                duration = get_audio_duration(file_path)

                # Calculate the number of slots and their duration
                num_slots = duration // 3
                if num_slots == 0:
                    continue  # Skip recordings that are too short
                total_duration = num_slots * 3
                
                # Load the audio file and compute its short-time energy
                y, sr = load_audio(file_path)
                slots = find_most_active_timeframe(y, sr, hop_length=512, total_duration=total_duration, slot_duration=3)
                
                # Determine is_stridor and is_phonatory from the folder structure and Excel sheet
                is_stridor, is_phonatory, status = determine_is_stridor_and_phonatory(parent_folder, root, patient_status_df)
                
                # Create the entry for the current recording
                entry = {
                    "uid": uid,
                    "name": parent_folder,
                    "recording": file_path,  # Include the full path
                    "recording_label": recording_label,
                    "is_stridor": is_stridor,
                    "is_phonatory": is_phonatory,
                    "status": status,
                    "duration": duration,
                    "recording_slots": slots
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
        if record['status'] == 0:
            no_stridor_count += 1
        elif record['status'] == 2:
            phonatory_count += 1
        elif record['status'] == 1:
            non_phonatory_count += 1
    
    return no_stridor_count, phonatory_count, non_phonatory_count


def filter_recordings_by_label(data, label_prefix):
    """
    Filter recordings to only include those with recording_label starting with the specified prefix.
    Exclude specific labels when the prefix is 'FIMO'.
    """
    exclude_labels = {"FIMO": ["FIMOcricoid", "FIMOthyroid", "FIMOC", "FIMOT"]}

    filtered_data = []
    for recording in data:
        if recording['recording_label'].startswith(label_prefix):
            if label_prefix == "FIMO" and recording['recording_label'] in exclude_labels["FIMO"]:
                continue
            filtered_data.append(recording)

    return filtered_data