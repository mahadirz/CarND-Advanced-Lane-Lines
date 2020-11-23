import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from itertools import permutations
import pickle
from scipy.ndimage.filters import uniform_filter1d


class LaneFinding:

    def __init__(self, dist_pickle_path):
        self.s_thresh = (170, 255)
        self.sx_thresh = (20, 100)
        self.dist_pickle = pickle.load(open(dist_pickle_path, "rb"))
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

        # vertices to select the Region of Interest
        # @todo dynamically determine this region
        self.vertices = np.array([[
            [280, 700],
            [595, 460],
            [725, 460],
            [1125, 700]
        ]], dtype=np.int32)

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

        # font
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        # font scale
        self.text_font_scale = 1.6
        # Red color in BGR
        self.text_color = (255, 255, 255)
        # Line thickness
        self.text_thickness = 3

        # track  the video related

        self.nframe = 0
        # video frame per second
        self.fps = 25
        self.left_fits = []
        self.left_fitx = []
        self.right_fits = []
        self.right_fitx = []

        # Metrices
        self.left_curv = []
        self.right_curv = []
        self.pos = []

        self.is_debug = False

    def reset(self):
        self.nframe = 0
        self.left_fits = []
        self.left_fitx = []
        self.right_fits = []
        self.right_fitx = []
        self.left_curv = []
        self.right_curv = []
        self.pos = []

    def undistort(self, img):
        """
        :param img:
        :return:
        """
        return cv2.undistort(
            img,
            self.dist_pickle['mtx'],
            self.dist_pickle['dist'],
            None, self.dist_pickle['mtx']
        )

    def binarize_image(self, img, selections=("L", "S")):
        """
        :param img:
        :param selections:
        :return:
        """
        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        options = {
            "H": hls[:, :, 0],
            "L": hls[:, :, 1],
            "S": hls[:, :, 2],

            "R": img[:, :, 0],
            "G": img[:, :, 1],
            "B": img[:, :, 2],
        }

        if "GR" in selections:
            options["GR"] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        first_channel = options[selections[0]]
        second_channel = options[selections[1]]

        # Sobel x
        sobelx = cv2.Sobel(first_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(second_channel)
        s_binary[(second_channel >= self.s_thresh[0]) & (second_channel <= self.s_thresh[1])] = 1

        # Stack each channel
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def warp_perspective(self, img):
        """
        Perform perspective Warp and return warped image and inverse M
        :param img:
        :return:
        """
        src = np.float32(self.vertices)
        dst = np.float32([
            [250, img.shape[0]],  # Bottom left
            [250, 0],  # Top left
            [1065, 0],  # Top right
            [1065, img.shape[0]]  # Bottom right
        ])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        ret, M_inv = cv2.invert(M)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        return warped, M_inv

    def hist(self, img):
        """
        :param img:
        :return:
        """
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0] // 2:, :]

        # i.e. the highest areas of vertical lines should be larger values
        histogram = bottom_half.sum(axis=0)

        return histogram

    def find_lane_pixels(self, binary_warped):
        """
        Find lane lines using histogram and sliding box
        :param binary_warped:
        :return:
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def draw_polynomial(self, binary_warped, ploty, leftx, lefty, rightx, righty, left_fitx, right_fitx):
        """
        Visualization
        Colors in the left and right lane regions

        :param binary_warped:
        :param ploty:
        :param leftx:
        :param lefty:
        :param rightx:
        :param righty:
        :param left_fitx:
        :param right_fitx:
        :return:
        """
        binary_warped[lefty, leftx] = [255, 0, 0]
        binary_warped[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return binary_warped

    def avg_line_fit(self, prev_fit, fit, nwindow=10):
        """
        :param prev_fit:
        :param fit:
        :param nwindow:
        :return:
        """
        prev_fit.append(fit)
        n = len(prev_fit)
        if n > 0:
            if n < nwindow:
                nwindow = n
            return np.array(prev_fit[n - nwindow:]).mean(axis=0)
        else:
            return fit

    def fit_polynomial(self, binary_warped, leftx, lefty, rightx, righty):
        """
        Can throw TypeError
        :param binary_warped:
        :param leftx:
        :param lefty:
        :param rightx:
        :param righty:
        :return:
        """
        left_fit = self.avg_line_fit(self.left_fits, np.polyfit(lefty, leftx, 2))
        right_fit = self.avg_line_fit(self.right_fits, np.polyfit(righty, rightx, 2))

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty, left_fit, right_fit

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        """
        :param binary_warped:
        :param left_fit:
        :param right_fit:
        :return:
        """
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # within the +/- margin of our polynomial function ###
        # Hint: consider the window areas for the similarly named variables ###
        # in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - self.margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                  left_fit[1] * nonzeroy + left_fit[
                                                                                      2] + self.margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - self.margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                    right_fit[1] * nonzeroy + right_fit[
                                                                                        2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return leftx, lefty, rightx, righty, out_img

    def visualize_search_poly(self, binary_warped, left_fitx, right_fitx, ploty):
        """
        :param binary_warped:
        :param left_fitx:
        :param right_fitx:
        :param ploty:
        :return:
        """
        ## Visualization ##
        # Generate a polygon to illustrate the search window area
        window_img = np.zeros_like(binary_warped)
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(binary_warped, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
        return result

    def unwarp(self, original_img, warped_binary, M_inv, left_fitx, right_fitx, ploty):
        """

        :param binary_warped:
        :param original_img:
        :param M_inv:
        :param left_fitx:
        :param right_fitx:
        :param ploty:
        :return:
        """
        # create a blank color warped
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                                 ploty])))])
        line_pts = np.hstack((left_line, right_line))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([line_pts]), (0, 255, 0))
        # Warp to original image
        new_warp = cv2.warpPerspective(color_warp, M_inv, (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, new_warp, 0.3, 0)
        return result

    def get_curvature_radius(self, ploty, leftx, rightx):
        """
        Calculates the curvature of polynomial functions in meters.
        :param ploty:
        :param leftx:
        :param rightx:
        :return:
        """
        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        leftx = leftx * self.xm_per_pix
        rightx = rightx * self.xm_per_pix
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, leftx, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, rightx, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        left_curverad = (
                                (1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
                        ) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                                 (1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
                         ) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def get_vehicle_position(self, imshape, left_fitx, right_fitx):
        """
        :param imshape:
        :param left_fitx:
        :param right_fitx:
        :return:
        """
        image_center = imshape[1] / 2
        # vehicle position is at the bottom where x at max index
        car_position = (left_fitx[-1] + right_fitx[-1]) / 2

        return (image_center - car_position) * self.xm_per_pix

    def add_num_estimation(self, img, curverad=0, pos_offset=0):
        """
        Display computed metric into the image
        :param img:
        :param left_curv:
        :param right_curv:
        :param pos_offset:
        :return:
        """
        # Reference from https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        img = img.copy()

        cv2.putText(img,
                    'Radius of Curvature = {0:,.2f} (m)'.format(curverad),
                    (60, 60),
                    self.text_font, self.text_font_scale, self.text_color, self.text_thickness
                    )

        pos_text = " "
        if pos_offset > 0:
            pos_text = " right "
        elif pos_offset < 0:
            pos_text = " left "
        cv2.putText(img,
                    'Vehicle is {0:,.2f}m{1}of center'.format(pos_offset, pos_text),
                    (60, 120),
                    self.text_font, self.text_font_scale, self.text_color, self.text_thickness
                    )

        if self.is_debug:
            cv2.putText(img,
                        'frame {}'.format(self.nframe),
                        (60, 180),
                        self.text_font, self.text_font_scale, (255, 0, 0), self.text_thickness
                        )

        return img

    @staticmethod
    def rolling_mean(x, n):
        """
        Compute rolling mean from array of x and n window
        :param x:
        :param n:
        :return:
        """
        if len(x) < n:
            n = len(x)
        return np.mean(x[len(x) - n:])

    def video_pipeline(self, input_image):
        """
        :param input_image:
        :return:
        """
        undistort_img = self.undistort(input_image)
        binary_image = self.binarize_image(undistort_img)
        warped_binary, M_inv = self.warp_perspective(binary_image)
        leftx, lefty, rightx, righty = [], [], [], []
        if len(self.left_fits) > 0 and len(self.right_fits) > 0:
            leftx, lefty, rightx, righty, out_img = self.search_around_poly(
                warped_binary, self.left_fits[-1], self.right_fits[-1]
            )
        min_conf_threshold = 9000
        if len(leftx) < min_conf_threshold and len(rightx) < min_conf_threshold and \
                len(lefty) < min_conf_threshold and len(righty) < min_conf_threshold:
            # search from scratch
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(warped_binary)

        left_fitx, right_fitx, ploty, left_fit, right_fit = self.fit_polynomial(out_img, leftx, lefty, rightx,
                                                                                righty)

        self.left_fitx.append(left_fitx)
        self.left_fits.append(left_fit)
        self.right_fitx.append(right_fitx)
        self.right_fits.append(right_fit)

        left_curverad, right_curverad = self.get_curvature_radius(ploty, left_fitx, right_fitx)
        if left_curverad and left_curverad > 0:
            self.left_curv.append(left_curverad)
        if right_curverad and right_curverad > 0:
            self.right_curv.append(right_curverad)

        pos = self.get_vehicle_position(warped_binary.shape, left_fitx, right_fitx)
        if pos:
            self.pos.append(pos)

        # unwarp and draw the filled lines
        output_image = self.unwarp(input_image, warped_binary, M_inv, left_fitx, right_fitx, ploty)

        # get running avg
        left_curv_avg = LaneFinding.rolling_mean(self.left_curv, 25)
        right_curv_avg = LaneFinding.rolling_mean(self.right_curv, 25)
        pos_avg = LaneFinding.rolling_mean(self.pos, 10)
        curverad = (left_curv_avg + right_curv_avg) / 2
        if curverad > 9000 and self.is_debug:
            # debug
            cv2.imwrite("debug/video_" + str(self.nframe) + ".jpg", cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        output_image = self.add_num_estimation(output_image, curverad, pos_avg)

        self.nframe += 1
        return output_image
