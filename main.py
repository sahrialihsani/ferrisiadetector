from io import StringIO
from pathlib import Path
import streamlit as st
import time
# import detect
from detect_image import main
import os
import sys
import argparse
from PIL import Image
from deep_list import *

hidemenu =  """
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:visible;
}
footer:after{
    content:'Copyright © 2022 Universitas Bengkulu';
    display:block;
    position:relative;
    color:tomato;
}
</style>
"""


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.markdown(hidemenu,unsafe_allow_html=True)
    st.title('Ferrisia Detector')

    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio(
     "Select input source",
     ('RTSP', 'Webcam', 'Local video', 'Local image'))

    conf_thres = st.sidebar.text_input("Class confidence threshold", "0.25")

    conf_thres_drift = st.sidebar.text_input("Class confidence threshold for drift dectection", "0.75")

    fps_drop_warn_thresh = st.sidebar.text_input("FPS drop warning threshold", "8")

    save_output_video = st.sidebar.radio("Save output video?",('Yes', 'No'))
    if save_output_video == 'Yes':
        nosave = False
        display_labels = False
    else:
        nosave = True
        display_labels = True
    
    save_poor_frame = st.sidebar.radio("Save poor performing frames?",('Yes', 'No'))
    if save_poor_frame == "Yes":
        save_poor_frame__ = True
    else:
        save_poor_frame__ = False
    
    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Local video":
        video = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)
        
        if st.sidebar.button("Start tracking"):
    
            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0")
                fps_warn = st.empty()
            
            with kpi2:
                st.markdown("**Detected objects in curret Frame**")
                kpi2_text = st.markdown("0")
            
            with kpi3:
                st.markdown("**Total Detected objects**")
                kpi3_text = st.markdown("0")
            
            # Updating System stats
            
            with js1:
                st.markdown("**Memory usage**")
                js1_text = st.markdown("0")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown("**Poor performing classes (Conf < {0})**".format(conf_thres_drift))
                inf_ov_1_text = st.markdown("0")
            
            with inf_ov_2:
                st.markdown("**No. of poor peforming frames**")
                inf_ov_2_text = st.markdown("0")
            
            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0")
            
            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0")

            detect(source=video.name, stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text, js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave, display_labels=display_labels, conf_thres_drift = float(conf_thres_drift), save_poor_frame__= save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text, inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn, fps_drop_warn_thresh = float(fps_drop_warn_thresh))

            inference_msg.success("Inference Complete!")

    # -------------------------- WEBCAM ----------------------------------
    if input_source == "Webcam":
        
        if st.sidebar.button("Start tracking"):

            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0")
                fps_warn = st.empty()
            
            with kpi2:
                st.markdown("**Detected objects in curret Frame**")
                kpi2_text = st.markdown("0")
            
            with kpi3:
                st.markdown("**Total Detected objects**")
                kpi3_text = st.markdown("0")
            
            # Updating System stats
            
            with js1:
                st.markdown("**Memory usage**")
                js1_text = st.markdown("0")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown("**Poor performing classes (Conf < {0})**".format(conf_thres_drift))
                inf_ov_1_text = st.markdown("0")
            
            with inf_ov_2:
                st.markdown("**No. of poor peforming frames**")
                inf_ov_2_text = st.markdown("0")
            
            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0")
            
            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0")

            detect(source='0', stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text, js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave, display_labels=display_labels, conf_thres_drift = float(conf_thres_drift), save_poor_frame__= save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text, inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn, fps_drop_warn_thresh = float(fps_drop_warn_thresh))
    
    # -------------------------- RTSP ------------------------------
    if input_source == "RTSP":
        
        rtsp_input = st.sidebar.text_input("IP Address", "rtsp://192.168.0.1")
        if st.sidebar.button("Start tracking"):
    
            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0")
                fps_warn = st.empty()
            
            with kpi2:
                st.markdown("**Detected objects in curret Frame**")
                kpi2_text = st.markdown("0")
            
            with kpi3:
                st.markdown("**Total Detected objects**")
                kpi3_text = st.markdown("0")
            
            # Updating System stats
            
            with js1:
                st.markdown("**Memory usage**")
                js1_text = st.markdown("0")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown("**Poor performing classes (Conf < {0})**".format(conf_thres_drift))
                inf_ov_1_text = st.markdown("0")
            
            with inf_ov_2:
                st.markdown("**No. of poor peforming frames**")
                inf_ov_2_text = st.markdown("0")
            
            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0")
            
            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0")

            detect(source=rtsp_input, stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text, js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave, display_labels=display_labels, conf_thres_drift = float(conf_thres_drift), save_poor_frame__= save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text, inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn, fps_drop_warn_thresh = float(fps_drop_warn_thresh))
    
    if input_source == "Local image":
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str,
                            default='data/images', help='source')
        # parser.add_argument('--img-size', type=int, default=640,
        #                     help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.485, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true',
                            help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true',
                            help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)


        uploaded_file = st.sidebar.file_uploader(
            "Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False


        if is_valid:
            print('valid')
            if st.button('Detect'):

                # detect(opt)
                main(opt)

                with st.spinner(text='Loading..'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                        #st.write(detect.s)
