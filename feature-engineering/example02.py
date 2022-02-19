
# %%
from pytesseract import image_to_string
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
# %%

##### Just a random picture from search
img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
img = requests.get(img)

img = Image.open(BytesIO(img.content))

# show image
img_arr = np.array(img)
plt.imshow(img_arr)
