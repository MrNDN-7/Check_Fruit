import streamlit as st
import pandas as pd
import numpy as np
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
import cv2
from PIL import Image




# @st.cache(suppress_st_warning=True)
# def get_fvalue(val):    
#     feature_dict = {"No":1,"Yes":2}    
#     for key,value in feature_dict.items():        
#         if val == key:            
#             return value 
# def get_value(val,my_dict):    
#     for key,value in my_dict.items():        
#         if val == key:            
#             return value
# app_mode = st.sidebar.selectbox('Select Page',['Home','Fruit']) #two pages

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
    
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
    
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

classes_file = "trai_cay.names"
classes = None
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')



class PredictImage():
    try:
        if st.session_state ["LoadModel"]==True:
            print ("Da Load  Model")
    except:
        st.session_state["LoadModel"]=True
        st.session_state["Net"]=cv2.dnn.readNet('yolov5_fruit.onnx')
        print('Load Model Lan Dau')

    def draw_label(im, label, x, y):
        """Draw text onto image at location."""
        # Get text size.
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
        # Display text inside the rectangle.
        cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
    def pre_process(input_image, net):
        # Create a 4D blob from a frame
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        # Set the input to the network
        net.setInput(blob)
        # Run the forward pass to get the output of the output layers
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs
    def post_process(input_image, outputs, classes):
        # Lists to hold respective values while unwrapping
        class_ids = []
        confidences = []
        boxes = []
        # Rows
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT
        # Iterate through detections
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                # Get the index of the max class score
                class_id = np.argmax(classes_scores)
                # Continue if the class score is above the threshold
                if classes_scores[class_id] > SCORE_THRESHOLD:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        # Perform non-maximum suppression to eliminate redundant, overlapping boxes with lower confidences
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # Draw bounding box
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
            # Class label
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            # Draw label
            PredictImage.draw_label(input_image, label, left, top)
        return input_image

def main():
    # def PredictImg():
        st.title('Nhận diện trái cây')
        uploaded_image = st.file_uploader('Upload File IMG', type=['jpg', 'png', 'jpeg'])
        frame = None
        # Load image.
        if uploaded_image is not None:
            frame = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
            modelWeights = "yolov5_fruit.onnx"
            net = cv2.dnn.readNet(modelWeights)
            # Process image.
            detections = PredictImage.pre_process(frame, net)
            img = PredictImage.post_process(frame.copy(), detections, classes)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
            if st.button("Predict"):
                t, _ = net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)
                st.write(label)

    # if app_mode == 'Fruit':
    #     PredictImg()
    
if __name__ == "__main__":
    main()

