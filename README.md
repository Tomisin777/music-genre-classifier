# Music Genre Classification with CNN

## Project Overview
This deep learning model classifies music into 10 genres using Convolutional Neural Networks (CNNs) and TensorFlow. The model analyzes mel spectrograms (visual representations of audio) to identify musical patterns unique to each genre.

## Dataset
- **Source:** GTZAN Dataset
- **Size:** 1,000 audio tracks (30 seconds each)
- **Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Split:** 800 training, 199 validation

## Methodology

### Audio Preprocessing
1. Loaded .wav audio files using librosa
2. Converted audio to mel spectrograms (visual representation)
3. Saved spectrograms as 128×128 pixel images

### CNN Architecture
- **Conv2D Layer:** 32 filters (3×3) - detects basic audio patterns
- **MaxPooling2D:** Reduces dimensionality
- **Conv2D Layer:** 64 filters (3×3) - detects complex patterns
- **MaxPooling2D:** Further reduction
- **Flatten:** Converts 2D to 1D
- **Dense Layer:** 128 neurons - combines patterns
- **Output Layer:** 10 neurons (softmax) - genre probabilities

**Total Parameters:** 7.4 million

## Results
- **Training Accuracy:** 97.4%
- **Validation Accuracy:** 39.2%

### Analysis
The model demonstrates significant overfitting, achieving high training accuracy but lower validation performance. This indicates the model memorized training examples rather than learning generalizable patterns.

**Contributing factors:**
- Small dataset (1,000 songs insufficient for 7.4M parameters)
- Genre ambiguity (overlap between similar genres like rock/metal)
- Model complexity relative to data size

**Still 4x better than random guessing (10%)!**

## Technologies Used
- **Python 3.13**
- **TensorFlow/Keras** - Neural network framework
- **Librosa** - Audio processing and feature extraction
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization

## Key Learnings
- CNNs can effectively process audio when converted to spectrograms
- Overfitting is a critical challenge with small datasets
- Deep learning models require substantial data to generalize
- Understanding the difference between training and validation metrics is crucial

## Future Improvements
1. **Data augmentation** - Generate more training examples
2. **Regularization** - Add dropout, L2 regularization
3. **Larger dataset** - Use FMA or Million Song Dataset
4. **Transfer learning** - Use pre-trained models
5. **Ensemble methods** - Combine multiple models
6. **Cross-validation** - More robust evaluation

## Project Structure
music-genre-classifier/
├── code/
│   ├── preprocess_audio.py
│   ├── create_spectrograms.py
│   └── train_cnn.py
├── Data/
│   ├── genres_original/
│   └── spectrograms/
├── README.md
└── requirements.txt

## How to Run
```bash
# Clone the repository
git clone [your-repo-url]
cd music-genre-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow librosa numpy pandas matplotlib scikit-learn seaborn

# Run preprocessing
python3 code/create_spectrograms.py

# Train model
python3 code/train_cnn.py

```


Author
Tomi - Biomedical Engineering Master's Student

Exploring the intersection of deep learning, audio processing, and machine learning

This project demonstrates practical application of CNNs for audio classification and highlights important concepts in deep learning such as overfitting, model architecture design, and the importance of sufficient training data.

