import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

def order_points(pts):
    # Coordinates :top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left = smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right = largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right = smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left = largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the result coordinates
    return rect.astype('int').tolist()



def find_dest(pts):
    (tl, tr, br, bl) = pts
    #maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    #maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    #result coordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def scan(img):
    # decide size of image 
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)


    # copy of original image
    orig_img = img.copy()

    #Repeat closing operation 
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

# ----------------Grabcut------------------------------------
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    #grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    #Find detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #largest detected
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    #detecting Edges through Contour 
    if len(page) == 0:
        return orig_img
    for c in page:
        
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        
        if len(corners) == 4:
            break

    #Sort the corners
    corners = sorted(np.concatenate(corners).tolist())

    
    corners = order_points(corners)

    destination_corners = find_dest(corners)

    #find homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    #transform with homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
    return final



def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Set title.
st.sidebar.title('OnScanner')

# Specify canvas parameters in application
uploaded_file = st.sidebar.file_uploader("Upload the img for Scanner:", type=["png", "jpg"])
image = None
final = None
col1, col2 = st.columns(2)

if uploaded_file is not None:

    #Converting the photo to opencv img
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    manual = st.sidebar.checkbox('Adjust Manually', False)
    h, w = image.shape[:2]
    h_, w_ = int(h * 400 / w), 400

    if manual:

        st.subheader('Select the 4 corners')
        st.markdown('### Double-Click to reset last point, Right-Click to select')

        #Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgb(90, 90, 90)",  # Fixed fill color with some opacity
            stroke_width=3,
            background_image=Image.open(uploaded_file).resize((h_, w_)),
            update_streamlit=True,
            height=h_,
            width=w_,
            drawing_mode='polygon',
            key="canvas",
        )
        st.sidebar.caption('Happy with the manual selection?')
        if st.sidebar.button('Get Scanned'):
            
            points = order_points([i[1:3] for i in canvas_result.json_data['objects'][0]['path'][:4]])
            points = np.multiply(points, w / 400)

            dest = find_dest(points)

            #find homography.
            M = cv2.getPerspectiveTransform(np.float32(points), np.float32(dest))

            #transform with homography.
            final = cv2.warpPerspective(image, M, (dest[2][0], dest[2][1]), flags=cv2.INTER_LINEAR)

            st.image(final, channels='BGR', use_column_width=True)
    else:
        with col1:
            st.title('Input')
            st.image(image, channels='BGR', use_column_width=True)
        with col2:
            st.title('Scanned')
            final = scan(image)
            st.image(final, channels='BGR', use_column_width=True)
    if final is not None:
        # Display link.
        result = Image.fromarray(final[:, :, ::-1])
        st.sidebar.markdown(get_image_download_link(result, 'output.png', 'Download ' + 'Output'),unsafe_allow_html=True)