# Language-detection-analyzer-model

This model analysis and identifies which Languages is written from the keyboard and virtually spoken !

**CODE**

import pandas as pd

import numpy as np

from langdetect import detect

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

print(data.head())

data["language"].value_counts()

x = np.array(data["Text"])

y = np.array(data["language"])

cv = CountVectorizer()

X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#using the Multinomial Naïve Bayesmodel

model=MultinomialNB()

model.fit(X_train,y_train)

model.score(X_test,y_test)

user = input("Enter a Text: ")

data = cv.transform([user]).toarray()

output = model.predict(data)

print(output)

def add_noise(audio_segment, gain):

num_samples = audio_segment.shape[0]

noise = gain * numpy.random.normal(size=num_samples)
    
return audio_segment + noise

def load_audio_file(audio_file_path):

audio_segment, _ = librosa.load(audio_file_path, sr=sample_rate)
    
return audio_segment

def spectrogram(audio_segment):
    
Compute Mel-scaled spectrogram image
    
hl = audio_segment.shape[0] // image_width
    
spec = librosa.feature.melspectrogram(audio_segment,
    
n_mels=image_height, 
                                     
hop_length=int(hl))

Logarithmic amplitudes
    
image = librosa.core.power_to_db(spec)
    

Convert to numpy matrix
    
image_np = numpy.asmatrix(image)

 Normalize and scale
    
 image_np_scaled_temp = (image_np - numpy.min(image_np))
    
  image_np_scaled = image_np_scaled_temp /
    
  numpy.max(image_np_scaled_temp)

  return image_np_scaled[:, 0:image_width]
