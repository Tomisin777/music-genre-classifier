import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

data_path = 'Data/genres_original'
output_path = 'Data/spectrograms'

#create output folder
os.makedirs(output_path, exist_ok=True)

genres = os.listdir(data_path)

#remove hidden files

genres = [g for g in genres if not g.startswith('.')]
print(f"Processing {len(genres)} genres...")

#Loop through each genre 
for genre in genres:
    print(f'Processing {genre}...')
    #create genre folder
    genre_path = os.path.join(output_path, genre)
    os.makedirs(genre_path, exist_ok=True)

    #Get all audio files
    genre_folder = os.path.join(data_path, genre)
    audio_files = [f for f in os.listdir(genre_folder) if  f.endswith('.wav')]

    #process each song 

    for i, audio_file in enumerate(audio_files):
        try:
            file_path = os. path.join(genre_folder, audio_file)
            audio, sr = librosa.load(file_path)

            #create spectrogram 
            spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            spec_db = librosa.power_to_db(spec, ref=np.max)

            #save as image 
            plt.figure(figsize=(10, 4))
            plt.axis('off')
            librosa.display.specshow(spec_db, sr=sr)

            #save 
            output_file = os.path.join(genre_path, f"{genre}_{i}.png")
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            if i % 10 == 0:
                print(f"  Processed {i}/{len(audio_files)} songs")
                
        except Exception as e:
            print(f"  Error with {audio_file}: {e}")
    
    print(f"✅ {genre} complete!")

print("🎉 All spectrograms created!")