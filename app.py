import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
import pyttsx3

p_time = 0

st.sidebar.title('Settings')
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLOv8', 'YOLOv7', 'YOLOv9', 'YOLOv10')
)

st.title(f'{model_type} Predictions')
sample_img = cv2.imread('logo2.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None


def speak(audio):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')

    engine.setProperty('voice', voices[1].id)
    engine.say(audio)
    engine.runAndWait()

if st.sidebar.checkbox('Load Model Options'):
    
    # YOLOv7 Model
    if model_type == 'YOLOv7':
        path_model_file = 'yolov7.pt'
        # GPU
        gpu_option = st.sidebar.radio(
            'PU Options:', ('CPU', 'GPU'))

        if not torch.cuda.is_available():
            st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
        else:
            st.sidebar.success(
                'GPU is Available on this Device, Choose GPU for the best performance',
                icon="✅"
            )
        # Model
        if gpu_option == 'CPU':
            model = custom(path_or_model=path_model_file)
        if gpu_option == 'GPU':
            model = custom(path_or_model=path_model_file, gpu=True)

    # YOLOv8 Model
    if model_type == 'YOLOv8':
        path_model_file = 'yolov8.pt'
        from ultralytics import YOLO
        model = YOLO(path_model_file)
    
    if model_type == 'YOLOv9':
        path_model_file = 'yolov9c.pt'
        from ultralytics import YOLO
        model = YOLO(path_model_file)
    if model_type == 'YOLOv10':
        st.caption("Work in Progress... >_<")
        # path_model_file = 'yolov10n.pt'
        # from ultralytics import YOLO
        # model = YOLO(path_model_file)

    # Load Class names
    class_labels = model.names

    # Inference Mode
    options = st.sidebar.radio(
        'Options:', ('Webcam', 'Image', 'Video'), index=1) # removed RTSP for now

    # Confidence
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=3
    )
    
    color_pick_list = [None]*len(class_labels)

if model_type in ['YOLOv8', 'YOLOv7']:
        # Image
        if options == 'Image':
            upload_img_file = st.sidebar.file_uploader(
                'Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')
                print('pred: ',pred)
                file_bytes = np.asarray(
                    bytearray(upload_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                FRAME_WINDOW.image(img, channels='BGR')

                if pred:
                    img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                    FRAME_WINDOW.image(img, channels='BGR')

                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with st.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq)
                        # print("🚀 ~ df_fq:", df_fq)

                        list_of_tuples = [(row.Class, row.Number) for row in df_fq.itertuples()]
                        print("🚀 ~ list_of_tuples:", list_of_tuples)

                        speak(f'This is what I have found {list_of_tuples}')
        
        # Video
        if options == 'Video':
            upload_video_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv'])
            if upload_video_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(upload_video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                # if pred:


        # Web-cam
        if options == 'Webcam':
            cam_options = st.sidebar.selectbox('Select Webcam Channel',
                                            ('0', '1', '2', '3'))
        
            if not cam_options == 'Select Channel':
                pred = st.checkbox(f'Predict Using {model_type}')
                cap = cv2.VideoCapture(int(cam_options))


        # RTSP
        if options == 'RTSP':
            rtsp_url = st.sidebar.text_input(
                'RTSP URL:',
                'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
            )
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(rtsp_url)

        # for i in range(len(class_labels)):
        #     classname = class_labels[i]
        #     color = color_picker_fn(classname, i)
        #     color_pick_list.append(color)

elif model_type in ['YOLOv9', 'YOLOv10']:
    print(1100110)

    # Image
    if options == 'Image':
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            pred = st.checkbox(f'Predict Using {model_type}')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')
            # st.caption(model(img)[0][0])

            if pred:
                def predict(model, imag, classes=[], conf=0.5):
                    if classes:
                        results = model.predict(imag, classes=classes, conf=conf)
                    else:
                        results = model.predict(imag, conf=conf)

                    return results

                def predict_and_detect(model, img, classes=[], conf=0.5, rectangle_thickness=draw_thick, text_scale=draw_thick, text_thickness=draw_thick):
                    results = predict(model, img, classes, conf=conf)
                    
                    # Initialize a Counter to keep track of class occurrences
                    class_counts = Counter()
                
                    for result in results:
                        for box in result.boxes:
                            # Update the counter with the class name
                            class_name = result.names[int(box.cls[0])]
                            class_counts[class_name] += 1
                            
                            # Draw the bounding box and label with a random color
                            color = tuple(np.random.randint(0, 255, size=3).tolist())
                            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
                            cv2.putText(img, f"{class_name}",
                                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, text_scale, color, text_thickness)
                    
                    # Convert the Counter to a DataFrame for easy viewing
                    df_fq = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Number'])
                    df_fq.index.name = 'Class'
                    
                    return img, df_fq
                
                img, df_fq = predict_and_detect(model, img, classes=[], conf=0.5)
                FRAME_WINDOW.image(img, channels='BGR')
                
                # Updating Inference results
                with st.container():
                    st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                    st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq)
                    # print("🚀 ~ df_fq:", df_fq)

                    list_of_tuples = [(row.Number, row.Index) for row in df_fq.itertuples()]

                    print("🚀 ~ list_of_tuples:", list_of_tuples)

                    speak(f'This is what I have found {list_of_tuples}')
    
    # Video
    # if options == 'Video':
    #     upload_video_file = st.sidebar.file_uploader(
    #         'Upload Video', type=['mp4', 'avi', 'mkv'])
    #     if upload_video_file is not None:
    #         pred = st.checkbox(f'Predict Using {model_type}')

    #         tfile = tempfile.NamedTemporaryFile(delete=False)
    #         tfile.write(upload_video_file.read())
    #         cap = cv2.VideoCapture(tfile.name)

            

if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
