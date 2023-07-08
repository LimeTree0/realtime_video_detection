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
# from ultralytics import YOLO


p_time = 0

st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLO Model', 'YOLOv8', 'YOLOv7')
)

st.title(f'{model_type} Predictions')
sample_img = cv2.imread('logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None

if not model_type == 'YOLO Model':
    path_model_file = st.sidebar.text_input(
        f'path to {model_type} Model:',
        f'eg: dir/{model_type}.pt'
    )
    if st.sidebar.checkbox('Load Model'):
        
        # YOLOv7 Model
        if model_type == 'YOLOv7':
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
            from ultralytics import YOLO
            model = YOLO(path_model_file)

        # Load Class names
        class_labels = model.names

        # Inference Mode
        options = st.sidebar.radio(
            'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        color_pick_list = []
        for i in range(len(class_labels)):
            classname = class_labels[i]
            color = color_picker_fn(classname, i)
            color_pick_list.append(color)

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
                        st.dataframe(df_fq, use_container_width=True)
        
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
            cam_options = st.sidebar.selectbox('Webcam Channel',
                                            ('Select Channel', '0', '1', '2', '3'))
        
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


if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()

    dangerous_level = {'person': 1, 'bicycle': 5, 'car': 5, 'motorcycle' : 5, 'airplane' : 5, 'bus' : 5,
                       'train': 5, 'truck': 5, 'boat': 5, 'traffic light': 4,'fire hydrant': 3, 'stop sign': 2,
                       'parking meter': 2, 'bench': 2, 'bird': 3, 'cat': 3, 'dog': 3, 'horse': 3, 'sheep': 3,
                       'cow': 5,'elephant': 5, 'bear': 5, 'zebra': 5, 'giraffe': 5, 'backpack': 1, 'umbrella': 1,
                       'handbag': 1, 'tie': 1, 'suitcase': 1, 'frisbee': 3, 'skis': 3,'snowboard': 3, 'sports ball': 3,
                       'kite': 1, 'baseball bat': 3, 'baseball glove': 3,'skateboard': 3, 'tennis racket': 3,
                       'bottle': 1, 'wine glass': 3, 'cup': 3, 'fork': 4, 'knife': 5, 'spoon': 1, 'bowl': 3, 'banana': 1, 'apple': 1,
                       'sandwich': 1, 'orange': 1,
                       'brocoli': 1, 'carrot': 1, 'hot dog': 1, 'pizza': 1, 'donut': 1, 'cake': 1, 'chair': 3, 'couch': 1,
                       'potted plant': 4, 'bed': 1, 'dining table': 1,
                       'toilet': 1, 'tv': 1, 'laptop': 1, 'mouse': 1, 'remote': 1, 'keyboard': 1, 'cell phone': 1,
                       'microwave': 1, 'oven': 5, 'toaster': 4, 'sink': 1,
                       'refrigerator': 1, 'book': 1, 'clock': 1, 'vase': 1, 'scissors': 1, 'teddy bear': 1,
                       'hair drier': 3, 'toothbrush': 1}
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
        df_fq['Level'] = [dangerous_level[Class] for Class in df_fq['Class']]
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
