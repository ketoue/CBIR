import cv2, os
from descriptor import glcm, bitdesc
import numpy as np


def extract_features(image_path, descriptor):
    img = cv2.imread(image_path, 0)
    if img is not None:
        features = descriptor(img)
        return features
    else:
        pass
descriptors = [glcm, bitdesc]
def process_datasets(root_folder):
    
    all_features_GLCM = [] # List to store all features and metadatas
    all_features_Bitdesc = [] # List to store all features and metadatas
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
                #GLCM
                features_glcm = glcm(image_rel_path)
                features_glcm  = features_glcm  + [folder_name, relative_path]
                all_features_GLCM.append(features_glcm )
                #BitDesc
                features_bitdesc = bitdesc(image_rel_path)
                features_bitdesc = features_bitdesc  + [folder_name, relative_path]
                all_features_Bitdesc.append( all_features_Bitdesc)


    #Glcm
    signaturesglcm = np.array(features_glcm)
    np.save('signatures_glcm.npy', signaturesglcm)

    #Bitdesc
    
    signaturesbitdesc = np.array(features_bitdesc)
    np.save('signatures__bitdesc.npy', signaturesbitdesc)

    print('Successfully stored!')
process_datasets('./image')