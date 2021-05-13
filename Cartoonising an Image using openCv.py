#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2

img = cv2.imread('siddhu.jpg')
# reads image as bgr(blue green red)


# In[15]:


from matplotlib import pyplot as plt
plt.imshow(img)
plt.show()


# In[16]:


# to plot this image in rgb i will use below function
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# change size of image to 10x10
plt.figure(figsize= (10,10))
plt.imshow(image)
plt.show()


# In[17]:


# now we apply Bilateral Filter on this, since Bilateral Filter is quite 
# a mathematically process, we reduce the size of image by 
# reducing the pixels
# using Pyramid Down technique-removing even rows and even columns

img_small = cv2.pyrDown(image)


# In[18]:


num_iter = 5
for i in range(num_iter):
    imag_small = cv2.bilateralFilter(img_small, d=9, sigmaColor = 9, sigmaSpace = 7)
# d is kernal size of 9x9......we apply bilateral filter 5 times


# In[19]:


img_rgb = cv2.pyrUp(img_small)


# In[20]:


plt.imshow(img_rgb)
plt.show()
# this image is more smooth than original image


# In[21]:


# # EDGE LINES using Adaptive Thresholding Image
# First we have to convert image to grayscale image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)
# 7 is kernal size


# In[22]:


img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
# any value above the threshhold will be 1 else 0


# In[23]:


plt.imshow(img_edge)
plt.show()


# In[24]:


img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)


# In[25]:


plt.imshow(img_edge)
plt.show()


# In[26]:


# now i will multiply both these images
array = cv2.bitwise_and(image, img_edge)


# In[27]:


plt.figure(figsize= (10, 10))
plt.imshow(array)
plt.show()


# In[ ]:




