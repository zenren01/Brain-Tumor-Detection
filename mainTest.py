import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.keras')

image=cv2.imread('D:\BRAIN TUMOR IMAGE CLASSIFICATION ML Project\pred/pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = np.argmax(model.predict(input_img), axis=-1)
print(result)




