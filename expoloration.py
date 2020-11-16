import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
"""Check 1 - See you can show image with anotations"""

img = np.array(Image.open('busesTrain/DSCF1013.JPG'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image (draw the image on the ax
ax.imshow(img)

# Create a Rectangle patch
rect = patches.Rectangle((1217, 1690), 489, 201, linewidth=1.5,
                         edgecolor='chartreuse', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
