import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def split_signal(signal, sample_rate, sec):
    """
    :param signal: raw signal which is going to be splitted
    :param sac: window length in seconds
    """
    windowLength = sample_rate*sec
    windowedSignal  = [] 
    for i in range(0,len(signal),windowLength):
        window = signal[i:i+windowLength]
        if len(window)>=windowLength:
            windowedSignal.append(window)
    return windowedSignal

def split_array_horizontaly(array, sliceWidth):
    if len(array.shape)==2:
        slicesCount = int(np.floor(sliceWidth.shape[1]/sliceWidth))
        return np.split(array[:,:slicesCount*sliceWidth],slicesCount,axis=1)
    elif len(array.shape)==1:
        slicesCount = int(np.floor(len(array)/sliceWidth))
        return np.split(array[:slicesCount*sliceWidth],slicesCount)

def get_emphasized_signal(signal, pre_emphasis=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def get_MFCC(signal, sample_rate=22050,num_mfcc=13, n_fft=2048, hop_length=512, scaled = False):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """
    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)
    if scaled:
        MFCCs = np.mean(MFCCs.T,axis=0)
    return MFCCs

def normalize_signal(data):
    '''
    Scale data in range [0, 1]
    Input: data
    '''    
    min_data = np.min(data)
    max_data = np.max(data)
    data = (data - min_data) / (max_data - min_data+1e-6)
    return data - 0.5

def add_padding_to_sound(sound,totalSamples=22050):
    soundLen = len(sound)
    idx = int(np.floor((totalSamples-soundLen)/2))
    padded = np.zeros(totalSamples)
    padded[idx:idx+soundLen] = sound
    return padded

def normalize_array(array):
    mn = np.min(array)
    mx = np.max(array)
    norm = (array - mn) * (1.0 / (mx - mn))
    return norm

def overlay_noise(sound, noise, noiseRate=0.05):
    soundNormalized = normalize_array(sound)
    noiseNormalized = normalize_array(noise)
    return ((1-noiseRate) * sound) + (noiseRate * noise)

def get_features(sound, preprocess_fun):
    """
    docstring
    """
    features = preprocess_fun(sound)
    normalized_ftrs = normalize_array(features)
    return normalized_ftrs

def sound_pipeline(sound, preprocess_fun, noise_fun = None, sample_rate=22050):
    """
    docstring
    """
    sound_len = len(sound)
    if sound_len==sample_rate:
        return get_features(sound, preprocess_fun)
    elif sound_len<sample_rate:
        padded = add_padding_to_sound(sound)
        return get_features(sound)

def sound_preprocessing(sound, noise_fun, sample_rate=22050):
    """
    docstring
    """
    sound_len = len(sound)
    if sound_len==sample_rate:
        return noise_fun(sound)
    elif sound_len<sample_rate:
        padded = add_padding_to_sound(sound, preprocess_fun)
        return noise_fun(sound)

def encode_labels(labels, label_encoder):
    # integer encode
    integer_encoded = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def invert_encode(y_hat, label_encoder):
    return label_encoder.inverse_transform([np.argmax(y_hat)])





