from cgi import test
from ctypes.wintypes import SIZE
import os
from pickle import TRUE
import numpy as np 
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas


model = load_model('save_fashion_mnist.h5')

SIZE = 300
mode  = st.checkbox("Draw (or Delete)?",True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=40,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')
# " id mode else transform "

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
    rescaled = cv2.resize(img,(SIZE,SIZE), interpolation=cv2.INTER_NEAREST)
    st.write("Model Input")
    st.image(rescaled)

if st.button ("predict"):
    test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1,28,28,1))
     
        
    if np.argmax(val[0]) == 0:
        st.write(f" Result -->0 Tishirt/Top")
    elif np.argmax(val[0]) == 1:
        st.write(f" Result --> 1 Trouser")
    elif np.argmax(val[0]) == 2:
        st.write(f" Result --> 2 Pullover")
    elif np.argmax(val[0]) == 3:
        st.write(f" Result --> 3 Dress")
    elif np.argmax(val[0]) == 4:
        st.write(f" Result --> 4 Coat")
    elif np.argmax(val[0]) == 5:
        st.write(f" Result --> 5 Sandal")
    elif np.argmax(val[0]) == 6:
        st.write(f" Result --> 6 Shirt")
    elif np.argmax(val[0]) == 7:
        st.write(f" Result --> 7 Sneaker")
    elif np.argmax(val[0]) == 8:
        st.write(f" Result --> 8 Bag")
    elif np.argmax(val[0]) == 9:
        st.write(f" Result --> 9 Ankle boot")
         
    # st.write(f"Result : {np.argmax(val[0])}")
    st.bar_chart(val[0])


    
    