#import libraries

import librosa
import numpy as np
import os

#Set path to data
data_path = 'Data/genres_original'

# Get list of all genres (folder names)
genres = os.listdir(data_path)
print(f"Genres found: {genres}")

# Load ONE audio file to test
# Pick the first blues song
test_file = os.path.join(data_path, 'blues', 'blues.00000.wav')

# Load the audio file
audio, sample_rate = librosa.load(test_file)

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sample_rate}")
print(f"Duration: {len(audio)/sample_rate} seconds")

#create spectrogram
import matplotlib.pyplot as plt

#creata mel spectrogram 
spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

#convert to decibels
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max )

print(f"Spectrogram shape: {spectrogram_db.shape}")

# Visualize it!
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram - Blues Song')
plt.tight_layout()
plt.savefig('example_spectrogram.png')
plt.show()

print("Spectrogram saved as example_spectrogram.png!")