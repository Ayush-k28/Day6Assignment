
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

st.title("Image Operations")
st.subheader("Choose the operation you want to perform from the sidebar")
st.write("This is a streamlit based application")
add_selectbox = st.sidebar.selectbox(
    "Types of Operations",
    ("About", "Face Recognition",'Face Detection','Selfie Segmentation')
)

if add_selectbox == "About":
    st.write("This application uses streamlit and opencv modules")


elif add_selectbox == "Face Detection":
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils 
    

    image_file_path = st.sidebar.file_uploader("Upload your image")
    image_train = face_recognition.load_image_file(image_file_path)
    image_encodings_train = face_recognition.face_encodings(image_train)[0]
    image_location_train = face_recognition.face_locations(image_train)[0]
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results:
                image_train = image
                cv2.rectangle(image_train, 
                    (image_location_train[3], image_location_train[0]),
                    (image_location_train[1], image_location_train[2]),
                    (0, 255, 0),
                    2)
                st.image(image) 
            else:
                st.write("Face not detected")


elif add_selectbox == "Face Recognition":
    image_file_path = st.sidebar.file_uploader("Upload your known image")
    image_file_path_2 = st.sidebar.file_uploader("Upload image you want to compare")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        image_train = face_recognition.load_image_file(image_file_path)
        image_encodings_train = face_recognition.face_encodings(image_train)[0]
        image_location_train = face_recognition.face_locations(image_train)[0]
        image_2 = np.array(Image.open(image_file_path_2))
        image_test = face_recognition.load_image_file(image_file_path_2)
        image_encodings_test = face_recognition.face_encodings(image_test)[0]
        results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
        dst = face_recognition.face_distance([image_encodings_train],image_encodings_test)
        if results:
            image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
            cv2.rectangle(image_train, 
                (image_location_train[3], image_location_train[0]),
                (image_location_train[1], image_location_train[2]),
                (0, 255, 0),
                2)
            cv2.putText(image_train,f"{results} {dst}",
                (60, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255,0),
                1)
            image_color =  cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
            st.image(image_color)
        else:
           st.write("Could not recognize the face.")


elif add_selectbox == "Selfie Segmentation":
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    colour = st.sidebar.radio("Choose the background colour for your image",
     ('blue', 'green', "red",'pink',"white"))
    if colour == "blue":
      BG_COLOR = (0,0, 255)
    elif colour == "green":
        BG_COLOR = (0,255, 0)
    elif colour == "pink":
        BG_COLOR = (255,228,196)
    elif colour == "red":
        BG_COLOR = (255,0,0)
    elif colour == "white":
        BG_COLOR = (255,255, 255)          

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
          image_file_path = st.sidebar.file_uploader("Upload your image")
          if image_file_path is not None:
              image = np.array(Image.open(image_file_path))
              st.sidebar.image(image)
              results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              
              condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
              
              bg_image = np.zeros(image.shape, dtype=np.uint8)
              bg_image[:] = BG_COLOR
              output_image = np.where(condition, image, bg_image)
              st.image(output_image)       
        





    
    

    





