import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from descriptor import glcm, bitdesc
# from descriptor import 
from distances import retrieve_similar_image
#from data_processing import extract_features
root_folder="./image"

for root, dirs, files in os.walk(root_folder):
        #print(root)
        for file in files:
            #print(file)
            if file.lower().endswith(('.jpg','.png', '.jpeg')):
                # Construct relative path
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                file_name = f'{relative_path.split("/")[0]}_{file}'
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                
 
# Load pre-computed featuresçq
glcm_features = np.load('signatures__bitdesc.npy', allow_pickle=True)
bit_features = np.load('signatures_glcm.npy', allow_pickle=True)
 
 
# Streamlit interface
st.title("Content-Based Image Retrieval")
uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png", "bmp", "tiff"])

num_results=0 
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR")
   
    descriptor = st.selectbox("Choose descriptor", ["GLCM", "BiT"])
    metric = st.selectbox("Choose distance metric", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])
    num_results = st.slider("Number of similar images to show", 1, 20, 5)
   
    if descriptor == "GLCM":
        feature = glcm(image)
        selected_features = glcm_features
    else:
        feature = bitdesc(image)
        selected_features = bit_features
   
    feature.append("query")
    sorted_indices, distances = retrieve_similar_image(feature, selected_features, metric.lower(),num_results)

if  num_results > 0:  
    st.write("Similar Images:")
    image_extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    for i in range(0, num_results, 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < num_results:
               idx = sorted_indices[i + j]
               name = selected_features[idx][-1]
               img_path = None
               for p in folder_name:
                  for ext in image_extensions:
                      potential_path = os.path.join(p, name + ext)
                      if os.path.exists(potential_path):
                        img_path = potential_path
                        break
                  if img_path:
                    break
               if img_path:
                  img = cv2.imread(img_path)
                  if img is None:
                    st.error(f"Error reading image {name} with extension {ext}")
                  else:
                    cols[j].image(img, caption=f"{name} - {metric} distance: {distances[i + j]:.2f}", use_column_width=True)
               else:
                st.error(f"Image {name} not found in any path with the specified extensions")














