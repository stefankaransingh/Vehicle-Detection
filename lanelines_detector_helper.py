import numpy as np
import cv2
import glob
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CAMERA_CALIBRATION_IMAGES_FOLDER = 'camera_cal'

def cvt_fig_to_img(fig):
    """
    The goal of this function is to convert a figure into an image that can be then saved.
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def calibrate_camera(calibration_folder=None,show_image=True):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibration_folder+'/calibration*.jpg')
    if show_image:
        fig=plt.figure(figsize=(10,70))
    num = 1
    gray = None

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        img_cp = np.copy(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if show_image:
                draw_img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                y = fig.add_subplot(17,2,num)
                y.set_title("Original Image",size=18)
                y.imshow(img_cp)

                num+=1

                x = fig.add_subplot(17,2,num)
                x.set_title("Chessboard Drawn with Corners",size=18)
                x.imshow(draw_img)
            num+=1
    if show_image:
        plt.show()
        fig.savefig("reference_images/finding-corners-images")

    return objpoints,imgpoints

def undistort_image(img,mtx,dist):
    """
    img: image to undistort
    mtx: Camera matrix
    dist: distortion coefficient
    """
    return cv2.undistort(img,mtx,dist,None,mtx)

def perspective_transform(img,img_size):
    """
    img: Source image
    src: Source points
    dst: Destination points
    """

    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                     [1250, 720],[40, 720]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped,Minv

def sobel(img=None,to_gray=False,sobel_kernel=3,x=None,y=None,thresh =(0,255)):
    img = np.copy(img)
    if to_gray:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    sobel = cv2.Sobel(img,cv2.CV_64F,x,y,ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary

def mag_thresh(img=None,to_gray=False,sobel_kernel=3,mag_thresh=(0,255)):
    img = np.copy(img)
    if to_gray:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    gradient_magnitude = np.sqrt(sobelx**2+sobely**2)
    scaled_factor = np.max(gradient_magnitude)/255
    gradient_magnitude = (gradient_magnitude/scaled_factor).astype(np.uint8)
    binary_output =np.zeros_like(gradient_magnitude)
    binary_output[(gradient_magnitude >= mag_thresh[0]) & (gradient_magnitude <= mag_thresh[1])] =1
    return binary_output

def dir_thresh(img=None,to_gray=False,sobel_kernel=3,thresh=(0,np.pi/2)):
    img = np.copy(img)
    if to_gray:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    absgraddir = np.arctan(np.absolute(sobely)/np.absolute(sobelx))
    binary_output=np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] =1
    return binary_output

def combine_thresh(img,to_gray=False,sobel_kernel=3):
    gradx = sobel(img,to_gray,sobel_kernel,1,0,(20,100))
    grady = sobel(img,to_gray,sobel_kernel,0,1,(20,100))
    mag_binary = mag_thresh(img,to_gray,sobel_kernel,(30,100))
    dir_binary = dir_thresh(img,to_gray,sobel_kernel=15,thresh=(0.7,1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

def seperate_color_channel(img,channel):
    if channel == 'r':
        return img[:,:,0]
    elif channel == 'g':
        return img[:,:,1]
    else:
        return img[:,:,2]

def seperate_hls_color_channel(img,channel):
    img_cvt = np.copy(img)
    img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_RGB2HLS)
    if channel == 'h':
        return img_cvt[:,:,0]
    elif channel == 'l':
        return img_cvt[:,:,1]
    elif channel == 's':
        return img_cvt[:,:,2]
    else:
        return img_cvt

def seperate_hsv_color_channel(img,channel):
    img_cvt = np.copy(img)
    img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_RGB2HSV)
    if channel == 'h':
        return img_cvt[:,:,0]
    elif channel == 's':
        return img_cvt[:,:,1]
    elif channel =='v':
        return img_cvt[:,:,2]
    else:
        return img_cvt

def seperate_luv_color_channel(img,channel):
    img_cvt = np.copy(img)
    img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_RGB2LUV)
    if channel == 'l':
        return img_cvt[:,:,0]
    elif channel == 'u':
        return img_cvt[:,:,1]
    elif channel =='v':
        return img_cvt[:,:,2]
    else:
        return img_cvt

def seperate_lab_color_channel(img,channel):
    img_cvt = np.copy(img)
    img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_RGB2Lab)
    if channel == 'l':
        return img_cvt[:,:,0]
    elif channel == 'a':
        return img_cvt[:,:,1]
    elif channel =='b':
        return img_cvt[:,:,2]
    else:
        return img_cvt

def color_thresh(img,thresh=(200,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def combine_color_thresh(img):
    r = color_thresh(seperate_color_channel(img,'r'),(200,255))
    s = color_thresh(seperate_hls_color_channel(img,'s'),(90,255))
    binary_output = np.zeros_like(r)
    binary_output[(r == 1) | (s ==1)] = 1
    return binary_output

def combine_thresh(img):
    r = color_thresh(seperate_color_channel(img,'r'),(200,255))
    s_binary = color_thresh(seperate_hls_color_channel(img,'s'),(90,255))
    l_binary = color_thresh(seperate_luv_color_channel(img,'l'),(215,255))
    b_binary = color_thresh(seperate_lab_color_channel(img,'b'),(145,200))
    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary ==1) |(b_binary==1) | (r == 1) | (s_binary ==1)]=1
    return combined_binary

def window_slide_fit_line_hist(img=None,nwindows = 9,do_window_search=True,left_fit=None,right_fit=None,show_img=False):
    histogram = np.sum(img[img.shape[0]//2:,:],axis=0)
    fig = None

    if show_img:
        fig=plt.figure(figsize=(12, 12))
        hst = fig.add_subplot(2,1,1)
        hst.set_title("Histogram")
        hst.plot(histogram)

    out_img = np.dstack((img,img,img))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    if do_window_search:
        window_height =np.int(img.shape[0]/nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100

    if not do_window_search and left_fit is not None and right_fit is not None:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    if do_window_search:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if show_img:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        result_plot = fig.add_subplot(2,1,2)
        result_plot.imshow(result)
        result_plot.plot(left_fitx, ploty, color='yellow')
        result_plot.plot(right_fitx, ploty, color='yellow')
        result_plot.set_xlim([0, 1280])
        result_plot.set_ylim([720, 0])
        result_plot.set_title("Window Slide Find Lane Line")
    if show_img:
        fig.savefig("reference_images/finding_lane_lines")
        output_img = cvt_fig_to_img(fig)
        return left_fit,right_fit,leftx,rightx,lefty,righty,output_img
    else:
        return left_fit,right_fit,leftx,rightx,lefty,righty

def measure_curvature(left_fit,right_fit,leftx,rightx,lefty,righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    #Measure Curvature in pixels
    left_curverad_px = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad_px = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad_m = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_m = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    return left_curverad_m,right_curverad_m,int((left_curverad_m + right_curverad_m)/2)

def center_of_vehicle(left_fit,right_fit):
    leftx_int = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
    rightx_int = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    # Calculate the position of the vehicle
    center = abs(640 - ((rightx_int+leftx_int)/2))
    return center

def draw(img,combined_binary,undist,Minv,left_fit,right_fit,center,curvature):


    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    """
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    """

    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))

    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    #cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (34,255, 34))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    if center < 640:
        text1 = 'Vehicle is {:.2f}m left of center'.format(center*3.7/700)
    else:
        text1 = 'Vehicle is {:.2f}m right of center'.format(center*3.7/700)
    text2 = 'Radius of curvature is {}m'.format(curvature)
    result = cv2.putText(result,text1, (150,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),6)
    result = cv2.putText(result,text2, (150,180), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),6)

    return result
