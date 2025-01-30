#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[3]:



from tensorflow.keras.utils import to_categorical

def load_dataset(file_path):
   
    data = pd.read_csv("fer2013.csv")
    
   
    X = []
    y = []

    for _, row in data.iterrows():
        pixels = np.fromstring(row['pixels'], sep=' ').reshape(48, 48)
        X.append(pixels)
        y.append(row['emotion'])

    X = np.array(X) / 255.0
    X = np.expand_dims(X, -1)

    y = np.array(y)

    return X, y

X, y = load_dataset('/mnt/data/fer2013.csv')
print(f"Original dataset shape: X={X.shape}, y={y.shape}")


# In[4]:


def filter_classes(X, y, target_count=3000):
   
    unique_classes = np.unique(y)
    filtered_X = []
    filtered_y = []


    for emotion in unique_classes:
    
        indices = np.where(y == emotion)[0]
        print(f"Class {emotion} has {len(indices)} images.")

        if len(indices) > target_count:
            selected_indices = np.random.choice(indices, target_count, replace=False)
        else:
            datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,

        filtered_X.append(X[selected_indices])
        filtered_y.append(y[selected_indices])


    filtered_X = np.concatenate(filtered_X, axis=0)
    filtered_y = np.concatenate(filtered_y, axis=0)

    filtered_X, filtered_y = shuffle(filtered_X, filtered_y, random_state=42)

    return filtered_X, filtered_y


X_filtered, y_filtered = filter_classes(X, y, target_count=3000)
print(f"Filtered dataset shape: X={X_filtered.shape}, y={y_filtered.shape}")


# In[5]:


def one_hot_encode_labels(y):
    
    return to_categorical(y)

y_filtered = one_hot_encode_labels(y_filtered)
print(f"One-hot encoded labels shape: y={y_filtered.shape}")


# In[6]:


def save_dataset(X, y, X_file, y_file):
    
    np.save(X_file, X)
    np.save(y_file, y)
    print(f"Dataset saved: {X_file}, {y_file}")


save_dataset(X_filtered, y_filtered, "X_filtered.npy", "y_filtered.npy")


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")


# In[8]:


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()


# In[16]:


model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

model.save('emotion_detection_model.h5')
print("Model training complete and saved as 'emotion_detection_model.h5'")


# In[17]:


model = tf.keras.models.load_model('emotion_detection_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_expression():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,/ 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Facial Expression Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_expression()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




