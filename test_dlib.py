import numpy as np
import pandas as pd 
import face_recognition 
import os 
import time

def get_encoding(path):
  image = face_recognition.load_image_file(path)
  face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
  if (len(face_locations) == 1):
    return np.array(face_recognition.face_encodings(image,face_locations))[0]
  return None

def parse_directory(dir):
  start = time.time()
  encodings = []
  count = 0
  for image in os.listdir(dir):
    path = os.path.join(dir,image)
    try:
      enc = get_encoding(path)
      if enc is not None:
        count += 1
        encodings.append(enc)
    except Exception as e:
      print(e)
  finish = time.time()
  print("Processed {} images. {} seconds, {} images/sec".format(count, finish-start, count / (finish - start)))
  return np.array(encodings)

data = './images/'
parse_directory(data)
