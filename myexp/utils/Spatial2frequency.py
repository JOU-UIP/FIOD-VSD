from PIL import Image
import numpy as np
import math
import cv2

# Open the image
img = cv2.imread("/home/underwater/Code/Data/DeepfishDomain/images/7117_Caranx_sexfasciatus_juvenile_f000000.jpg")

# Get the dimensions of the image
height,width, c= img.shape

# Calculate the new dimensions that are multiples of 8
cu=math.ceil(width / 8)
cv=math.ceil(height / 8)
new_width = cu * 8
new_height = cv * 8

# Resize the image to the new dimensions
img = cv2.resize(img,(new_width, new_height))

dct_array=np.zeros((3,64,new_height,new_width),dtype=np.float32)
for t,img in enumerate(cv2.split(img)):
    img_array=np.array(img, dtype=np.float32)
    dct_array[t,:,:] = np.expand_dims(cv2.dct(img_array),0).repeat(64,axis=0)
    s=0
    for i in range(8):
        for j in range(8):
            dct_array[t,s,i*cu:(i+1)*cu,j*cv:(j+1)*cv]=0
            cv2.imwrite(f'output/dct_{t}_{s}.jpg',dct_array[t,s,:])
            s+=1

dct_array=np.transpose(dct_array,(1,0,2,3))
for k in range(64):
    inv_dct_array=[np.clip(cv2.idct(inv_dct), 0, 255).astype(np.uint8) for inv_dct in dct_array[k]]
    img_merged = cv2.merge(inv_dct_array)
    print(np.max(img_merged),np.min(img_merged))
    cv2.imwrite(f"output/output_{k}.jpg",img_merged)
