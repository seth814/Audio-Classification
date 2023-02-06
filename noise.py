import numpy as np
import os
import librosa
from tqdm import tqdm
import soundfile as sf

def shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def manipulate(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def augment(x=50):
    for file in tqdm(os.listdir('original_data/')):
        if os.path.isfile(file):
            continue
        i = 0
        for f in os.listdir('original_data/' + file):
            print(f)
            p = os.path.join('original_data/', file) + '/' + f
            data, sr = librosa.core.load(p)
            for j in range(x):
                d = manipulate(data, 0.05)
                d1, d2 = shift(data, sr, 3, 'both'), shift(d, sr, 3, 'both')
                p = f'augmented_data/{file}'
                if os.path.exists(p) is False:
                    os.mkdir(p)
                sf.write(f'{p}/{i}_{1}.wav', d1, sr)
                sf.write(f'{p}/{i}_{2}.wav', d2, sr)
                sf.write(f'{p}/{i}_{0}.wav', d, sr)
                i += 1

if __name__ == "__main__":
    augment()


