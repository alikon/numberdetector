# Import the modules
import os
import sys
from glob import glob
import time
import cv2
#from sklearn.externals import joblib
#import joblib
from skimage.feature import hog
import sklearn
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
# Get the path of the training set

def detectNumber(image):
    classifier = 'digits_cls.pkl'
    #image = 'example1.jpg'

    # Load the classifier
    clf, pp = joblib.load(classifier)

    # Read the input image 
    im = cv2.imread(image)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        nbr = clf.predict(roi_hog_fd)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()

DEMO_IMAGE = "./images/example1.jpg"
sys.path.insert(0, ".")
st.write("""
    # Number prediction
    """)
# provide options to either select an image form the gallery, upload one, or fetch from URL
gallery_tab, upload_tab, url_tab = st.tabs(["Gallery", "Upload", "Image URL"])

with gallery_tab:
    gallery_files = glob(os.path.join(".", "images", "*"))
    gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("-", " "): image_path
        for image_path in gallery_files}

    options = list(gallery_dict.keys())
    file_name = st.selectbox("Select Art", 
                        options=options, index=options.index("example1"))
    file = gallery_dict[file_name]
    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")
    image = Image.open(file)

with upload_tab:
    st.write("## Upload a picture that contains a face")

    uploaded_file = st.file_uploader("Choose a file:")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
   # else:
   #     image = Image.open(DEMO_IMAGE)
with url_tab:
    url_text = st.empty()
    url_reset = st.button("Clear URL", key="url_reset")
    if url_reset and "image_url" in st.session_state:
        st.session_state["image_url"] = ""
        st.write(st.session_state["image_url"])

    url = url_text.text_input("Image URL", key="image_url")
    
    if url!="":
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

start = time.time()
detectNumber(image)

col1, col2 = st.columns(2)
with col1:
    st.header("Original")
    st.image(image)
#with col2:
#    st.header("Model")
#    st.image(frameFace)
#with st.expander("See explanation"):
#    for mesg in msg:
#        st.write(mesg)

end = time.time()
st.write(f"Done in {round(end - start, 3)}")