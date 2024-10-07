import os
import pyedflib
import numpy as np
import re
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA

# Function to apply band-pass filtering
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

# Function to normalize the signal
def normalize(data):
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

# Function to perform ICA
def perform_ica(signals, max_iter=1000, tol=0.001):
    ica = FastICA(n_components=signals.shape[0], max_iter=max_iter, tol=tol, random_state=0)
    components = ica.fit_transform(signals.T).T
    return components, ica

# Function to reconstruct the signal after removing artifacts
def reconstruct_signal(components, ica, components_to_remove):
    components[components_to_remove, :] = 0
    reconstructed_signal = ica.inverse_transform(components.T).T
    return reconstructed_signal

# Function to apply notch filter for 50-60 Hz noise removal
def apply_notch_filter(signal, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    filtered_signal = filtfilt(b, a, signal, axis=1)
    return filtered_signal

# Function to compute band power in a specified frequency band
def compute_band_power(data, fs=160, band=(8, 12), nperseg=128):
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    band_power = np.trapz(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# Updated function to map event markers to labels based on the run number
def map_labels(event_type, run_number):
    if run_number in [1, 2]:
        return 'Rest'
    elif run_number in [3, 7, 11]:  # Real movement runs for left/right fist
        return 'Real Left Fist' if event_type == 1 else 'Real Right Fist'
    elif run_number in [4, 8, 12]:  # Imaginary movement runs for left/right fist
        return 'Imaginary Left Fist' if event_type == 1 else 'Imaginary Right Fist'
    elif run_number in [5, 9, 13]:  # Real movement runs for both fists/feet
        return 'Real Both Fists' if event_type == 1 else 'Real Both Feet'
    elif run_number in [6, 10, 14]:  # Imaginary movement runs for both fists/feet
        return 'Imaginary Both Fists' if event_type == 1 else 'Imaginary Both Feet'
    else:
        return 'Unknown'

# Updated function to extract event markers from the .edf file
def extract_events(edf_reader):
    annotations = edf_reader.readAnnotations()
    event_types = []
    for annotation in annotations:
        if len(annotation) > 2:
            description = annotation[2]
            if isinstance(description, str):  # Ensure it's a string
                if 'T0' in description:
                    event_type = 0  # Rest
                elif 'T1' in description:
                    event_type = 1  # Left fist or both fists
                elif 'T2' in description:
                    event_type = 2  # Right fist or both feet
                else:
                    event_type = -1  # Unknown
            else:
                event_type = -1  # Handle cases where the description isn't a string
        else:
            event_type = -1  # Handle cases where the annotation doesn't have enough elements
        event_types.append(event_type)
    return event_types

# Set the base path to the folder containing all subject folders
base_data_path = '/home/group3/Downloads/eeg-motor-movementimagery-dataset-1.0.0/files'

# Initialize an empty list to store the extracted features
export_data = []

# Iterate over each subject folder (from S001 to S109)
for subject_id in range(1, 110):  # Range is from 1 to 109 inclusive
    subject_folder = f'S{str(subject_id).zfill(3)}'
    subject_path = os.path.join(base_data_path, subject_folder)
    
    if not os.path.exists(subject_path):
        print(f"Subject folder {subject_folder} does not exist. Skipping...")
        continue
    
    # Iterate over each .edf file in the subject's folder
    for file_name in os.listdir(subject_path):
        if file_name.endswith('.edf'):
            edf_file = os.path.join(subject_path, file_name)
            
            # Extract run number from the filename
            run_match = re.search(r'R(\d+)', file_name)
            if run_match:
                run_number = int(run_match.group(1))
                print(f"Processing file: {file_name} - Run Number: {run_number}")
            else:
                run_number = None  # Handle cases where the run number is not found
                print(f"Run number not found in file name: {file_name}. Skipping this file.")
                continue
            
            # Reading the EEG data
            f = pyedflib.EdfReader(edf_file)
            n = f.signals_in_file
            signals = np.zeros((n, f.getNSamples()[0]))
            for i in np.arange(n):
                signals[i, :] = f.readSignal(i)
            
            # Extract event markers
            event_types = extract_events(f)
            
            # Keep track of processed event types to avoid duplicates
            processed_events = set()

            for event_type in event_types:
                label = map_labels(event_type, run_number)
                
                # Skip if this event_type and run_number combination was already processed
                if (event_type, run_number) in processed_events:
                    continue
                
                # Add the current event to the processed set
                processed_events.add((event_type, run_number))
                
                # Preprocessing steps
                signal_freq = 160  # Ensure this is the correct sampling frequency for your data
                filtered_signals = bandpass_filter(signals, 0.5, 40, signal_freq)
                normalized_signals = normalize(filtered_signals)
                
                # Perform ICA
                components, ica = perform_ica(normalized_signals, max_iter=1000, tol=0.001)
                
                # Identify and remove artifacts based on kurtosis
                kurtosis_values = kurtosis(components, axis=1)
                components_to_remove = np.where(kurtosis_values > 10)[0]
                
                # Reconstruct the signal without the artifact components
                cleaned_signal = reconstruct_signal(components, ica, components_to_remove)
                
                # Apply notch filter to remove 50-60 Hz noise
                cleaned_signal_notch_filtered = apply_notch_filter(cleaned_signal, fs=160, freq=50.0, Q=30.0)
                
                # Feature extraction
                segment_mean = np.mean(cleaned_signal_notch_filtered, axis=1)
                segment_variance = np.var(cleaned_signal_notch_filtered, axis=1)
                segment_skewness = skew(cleaned_signal_notch_filtered, axis=1)
                segment_kurtosis = kurtosis(cleaned_signal_notch_filtered, axis=1)
                alpha_power = compute_band_power(cleaned_signal_notch_filtered[0, :], fs=160, band=(8, 12), nperseg=128)
                beta_power = compute_band_power(cleaned_signal_notch_filtered[0, :], fs=160, band=(12, 30), nperseg=128)

                # Create a list of the extracted features and include the label
                segment_data = list(segment_mean) + list(segment_variance) + list(segment_skewness) + list(segment_kurtosis) + [alpha_power, beta_power, label]
                export_data.append(segment_data)

# Define column names for the dataframe
n_channels = len(segment_mean)
columns = [f'Mean_Channel_{i+1}' for i in range(n_channels)] + \
          [f'Variance_Channel_{i+1}' for i in range(n_channels)] + \
          [f'Skewness_Channel_{i+1}' for i in range(n_channels)] + \
          [f'Kurtosis_Channel_{i+1}' for i in range(n_channels)] + \
          ['Alpha_Power', 'Beta_Power', 'Target']

# Create a pandas dataframe and save it as a CSV file
df_export = pd.DataFrame(export_data, columns=columns)
output_file_path = 'extracted_features_all_subjects.csv'
df_export.to_csv(output_file_path, index=False)

print("Data processing and export completed.")
