import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

from PIL import ImageGrab
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
#from directkeys import PressKey, W, A, S, D
#from statistics import mean

# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
#grab screen from partial area of windows
#screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
# convert to gray
#gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
# edge detection
#edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#mask = np.zeros_like(edges)
ignore_mask_color = 255
#cv2.fillPoly(mask, vertices, ignore_mask_color)
#masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments
#line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
#lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)


def process_img(image):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    original_image = image
    # convert to gray
    processed_img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection
    processed_img_Canny = cv2.Canny(processed_img1, low_threshold, high_threshold)

    #processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0) #sentdex
    processed_img2 = cv2.GaussianBlur(processed_img_Canny,(kernel_size, kernel_size),0) #udacity
    #processed_img3 = cv2.Canny(processed_img2, low_threshold, high_threshold)

    imshape = image.shape
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    #udacity의 정지이미지 검출에서 가져온 영역설정 -- 아래것과 영역비교해볼것
    vertices = np.array([[10,650],[10,600],[450,400],[850,400],[1280,650]], np.int32)
    #sentdex의 python plays gta5 6강에서 가져온 영역 (위의 것과 결과비교해볼것)-이것은 지평선 아래만 인식
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    #udacity는 딱히 영역을 정하고 하지 않아서 vertices가 없는거 같다. photo 검출에서는 np.array가 있음,윗줄 참조

    #여기에 try except(sentdex)의 것을 주석으로 남겨두면 invalid syntax 에러가 난다 그래서 주석도 지웠음

    processed_img = roi(processed_img2, [vertices]) #가우시안 블러와 차선영역(사다리꼴)만 설정해서 processed_img에 건네줌

    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,       15) #sentdex
    #lines = cv2.HoughLinesP(processed_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap) #udacity
    #line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) #udacity
    #draw_lines(line_img, lines)
    return processed_img,original_image
    #processed_img는 가우시안 불러까지만 적용되어 있음, canny는 아직 미적용






def roi(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(image,img, lines, color=[0, 0, 255], thickness=6):
    #def draw_lines(img, lines, color=[255, 0, 0], thickness=2): #udacity코드
#def draw_lanes(img, lines, color=[0, 255, 255], thickness=3): #sentdex의 코드
    """
    `edges` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    cv2.imshow(lines_edges)


def weighted_img(processed_img, original_image, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(original_image, α, img, β, γ)


def main():
    last_time = time.time()
    while True:
        screen =  np.array(ImageGrab.grab(bbox=(0,40, 800,600)))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image = process_img(screen) #def process_img함수에 screen영역을 전달한 후 그 영역을 2개의 변수에게 전달
        cv2.namedWindow('02_file', cv2.WINDOW_NORMAL) #내가 추가한 것
        cv2.resizeWindow("02_file", 640,480)
        cv2.imshow('02_file', new_screen)

        cv2.namedWindow('02_file_01', cv2.WINDOW_NORMAL) #내가 추가한 것
        cv2.imshow('02_file_01',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



'''
Canny_edge = cv2.Canny(processed_img, low_threshold, high_threshold)

cv2.namedWindow('Gaussian_blur', cv2.WINDOW_NORMAL)
cv2.imshow('Gaussian_blur', new_screen) #new_screen을 window라는 이름으로 창을 띠워줌,process_img의 기능을 응용한 것이니 가우시안 불러까지 된 듯 함
                                 #Canny edge가 아직 미적용 상태
cv2.namedWindow('Canny_edge',cv2.WINDOW_NORMAL)
cv2.imshow('Canny_edge',Canny_edge,Cannyed_image)
'''



main()

import udacity_photo_line_detection_refresh.py as udacity_photo_refresh
udacity_photo_refresh.edge_detection()

#이 파일(2번)은 칼라,흑백 영상이 뜨는 기본적인 파일이다. 흑백영상은 원본 영상에서 차선만 추출하게 설정이 되어 있다






'''
while을 1로 바꾸고 k변수를 도입해서 사용하자 while문이 작동하기 시작했다
https://answers.opencv.org/question/181837/how-to-refresh-the-image-content-in-a-window/
위의 페이지를 참조했다
'''

'''

screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
vertices = np.array([[10,650],[10,600],[450,400],[850,400],[1280,650]], np.int32)

screen의 영역에서
vertices의 영역만 ROI로 선정하는 것이다
그리고 Udacity의 photo에서 edge를 추출하는 것을 while문을 이용해서 무한 반복시키면 어떨까?

또한 아래 방법을 이용하면 직접 읽어들여서 굳이 화면 한 구석을 차지하지 않게 만들 수 있을 것이다.


https://zzsza.github.io/data/2018/01/23/opencv-1/
파일로 비디오 읽기 (Video Read)
파일을 직접 읽기도 정말 간단합니다

import cv2

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''
