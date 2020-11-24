# Udacity Self Driving Car, 2nd Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[video1]: ./project_video_output.mp4 "Video"
[img_undistorted]: ./writeup_images/undistorted.png "Undistorted"
[img_lane_undistorted]: ./writeup_images/lane_distortion.png "lane_Undistorted"
[img_RGB_HLS_GR]: ./writeup_images/RGB_HLS_Gray.png "RGB HLS Gray comparison"
[img_permutation]: ./writeup_images/permutation.png "Permutation comparison"
[img_straight_warped]: ./writeup_images/straight_warped.png "Straight Warped"
[img_hist]: ./writeup_images/hist.png "Histogram"
[img_search_box]: ./writeup_images/search_box.png "Search boxes"
[img_search_poly]: ./writeup_images/search_poly.png "Search poly"
[img_unwarp]: ./writeup_images/unwarp.png "Unwarp"
[img_final]: ./writeup_images/final.png "final"
[img_test_images]: ./writeup_images/all_test_images.png "img_test_images"
[img_anomaly_1]: ./writeup_images/anomaly_1.png "anomaly_1"
[img_anomaly_2]: ./writeup_images/anomaly_2.png "anomaly_2"

### Project Structures

* [`camera calibration.ipynb`](<camera calibration.ipynb>) is the notebook for creating the calibration matrix.
* [`LaneFinding.py`](LaneFinding.py) stores the class for line finding code
* [`Lane Finding Prototyping.ipynb`](<Lane Finding Prototyping.ipynb>) Example of LaneFinding class usage
* `calibration.pickle` is the saved pickle file for the calibration matrix
* [`project_video_output.mp4`](project_video_output.mp4) the vide output for the project
* `output_images` a the folder that contains image output for the project test images
* `debug` a folder that contains debugging image frame and etc 

### Camera Calibration

Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image
example of such distortion object may appear closer or further away than they actually are. This preliminary stage is
to find the camera matrix and distortion coefficients to be used in further stages to correct the image.

According to OpenCV python documentation, at least 10 test pattern images are required to get the calibration data [[1]](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html).
In this project, 20 images of 9x6 chessboard are provided inside the `camera_cal` folder. 
The full steps of the calibration can be found in the [`camera calibration.ipynb`](<camera calibration.ipynb>)  codescalibration as such:

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
```

`objpoints` is the list of the corners in 3D points in real world, where (0,0,0) is the origin until (8,5,0). 
`imgpoints` is the list of the 2D corners of in the image plane. `cv2.findChessboardCorners` is used to get this points.

The  camera matrix `mtx` and distortion coefficient `dist` is then used to undistort the image.

```python
cv2.undistort(image, mtx, dist, None, mtx)
```

The following image shows the before and after the un-distortion.

![alt text][img_undistorted]

The values are then saved into `calibration.pickle` to be reused in next stage.

### Pipeline (single images)

I've written a new python class file in [`LaneFinding.py`](LaneFinding.py) to hold the full implementation of the Lane Finding.
The step by step of the prototyping can be found in the [`camera calibration.ipynb`](<camera calibration.ipynb>) that shows the
usage of the class as well as description of every stages. 

The class can be initialize as so. The only required parameter is the path to the camera calibration file.

```python
# import the class
from LaneFinding import LaneFinding

lane_finding = LaneFinding("calibration.pickle")
```

#### 1. Correction for Distortion

Before any manipulation is carried out, the image is corrected for distortion using the matrix and coefficient that we computed
from camera calibration step. It's assume here the same camera is used to record the video.

The image below shows the before and after the un-distorted on `test1.jpg`. If inspected closely there are small detail differences in the image
especially on the bottom section.
 
![alt text][img_lane_undistorted]

#### 2. Color transform and thresholded binary image

In this step, the objective is to make the lane's lines more contrast in relative to road. 
The output will a binary black and white and should ideally contains the lines without any noises under varying degrees of daylight and shadow. 

There are two colors of the line, which is yellow and white. In order to determine which color that may best detect these colors, 
the individual RGB and HLS color channels are plotted as such to compare visually:

![alt text][img_RGB_HLS_GR]

S image does visually look more robust to detect the lines under extreme contribution compare to the rest.

The next step is to apply gradient and color threshold onto the image. `binarize_image` is the method that contains the implementation, it 
accepts the image as first parameter and the list of size two for the color channels.

```python
# the allowable selections: R,G,B,H,L,S,GR
binarize_image(img, selections=("L", "S"))
```

The first selection is applied for gradient thresholding while the second selection only applied for color thresholding.
For the color threshold I use (170,255) while for gradient one is (20, 100) which is the default value taken from the Udacity's lesson.
The output from this method is a binary image of 0 and 1.

```
self.s_thresh = (170, 255)
self.sx_thresh = (20, 100)
```

To find out the best channel's combination, python permutation is used here to generate the list and
write the image into debug folder for manual inspect.

```python
selections = ["GR", "H", "L", "S","R","G","B"]
perm = permutations(selections, 2) 
image = mpimg.imread("test_images/test1.jpg")
perm = list(perm)
for s in selections:
    perm.append((s,s))
for i,p in enumerate(list(perm)):  
    result = lane_finding.binarize_image(image, selections=[p[0],p[1]]) 
    plt.imsave(f"debug/{p[0]}_{p[1]}.jpg",result, cmap="gray")
```

The best 9 images from the permutation is displayed below. 

![alt text][img_permutation]

Despite the individual channel seems perform quite bad except for the S channel,
the combination of two produce better result. The S+S and H+S is probably the best combination for this image.

For the rest of the stages, I've used H+S as its perform slightly better from my tests.

#### 3. Perspective Transform and Warp

Once the image is binarized with a good filters, the perspective warping is performed on the region around the lines.
The region is a four points in a trapezoidal shape. 

The method that perform this transformation can be located in LaneFinding class under

```python
def warp_perspective(self, img, src=None, dst=None):
```

It accepts a binary image and two optional parameters of src and dst. I've experimented with several hard coded value of these
parameters and found the following points are the best for my implementation.

```python
vertices = np.array([[
             [209,  720],  # Bottom left
             [579,    460],  # Top left
             [698,   460],  # Top right
             [1115, 720] # Bottom right
        ]], dtype=np.int32)
src = vertices.reshape([4,2]).astype('float32')
dst = np.float32([
             [250,  720],  # Bottom left
             [250,    0],  # Top left
             [1100,   0],  # Top right
             [1100, 720] # Bottom right
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 209, 720      | 250, 720      | 
| 579, 460      | 250, 0        |
| 698, 460      | 1100, 0       |
| 1115, 720     | 1100, 720     |

From visual inspection using one the straight line example image, the warped image contains a bird's eyes of a roughly two
perpendicular lines.

![alt text][img_straight_warped]

#### 4. Identify lane-line pixels and fit their positions with a polynomial

Two methods are used to detect the lane-line pixel namely using sliding boxes and polynomial line from previous frame.

The Sliding boxes method is implemented in `find_lane_pixels`. It utilizes a box of size 200x80, the starting point of the box is determined 
using the highest sum of vertical pixels from left and right of it's bottom half histogram. 

```python
# Create histogram of image binary activations
histogram = lane_finding.hist(warped_binary)
```

![alt text][img_hist]

The two point is the center of the box, the next step is to locate the the non-zero pixels inside the box and append these
pixels indices into a list. The center of the next window of the box is adjusted according to the average x coordinates
of the found pixels. The operations are continued until reached to the top of the image. 

The `find_lane_pixels` will return the position of left and right of the found pixels. These coordinates then feed
into `fit_polynomial` method to compute the second order polynomial coefficients by using function `polyfit` from numpy.
These coefficients is used to find the x coordinate of the left and right line by fitting into second degree polynomial
equation, f(y)=Ay2+By+C.

The following snippet is an example of the implementing the methods.

```python
leftx, lefty, rightx, righty, out_img, rects = lane_finding.find_lane_pixels(warped_binary)

left_fitx, right_fitx, ploty, left_fit,right_fit = lane_finding.fit_polynomial(
    out_img, leftx, lefty, rightx, righty
)

out_img = lane_finding.draw_search_sliding(
    out_img, rects, ploty, leftx, lefty, rightx, righty,left_fitx, right_fitx 
)

plt.imshow(out_img)
plt.show()
```

The search can be illustrated as below:

![alt text][img_search_box]

The second method is search around the polynomial line from previous frame, this method is more efficient compared
to sliding boxes because of no loop involved, it is preferred by default on video processing.

The following codes demonstrate the approach in high level. 

```python
leftx, lefty, rightx, righty, out_img = lane_finding.search_around_poly(warped_binary, left_fit, right_fit)

left_fitx, right_fitx, ploty, left_fit,right_fit = lane_finding.fit_polynomial(out_img, leftx, lefty, rightx, righty)

result = lane_finding.draw_search_poly(out_img, ploty, leftx, lefty, rightx, righty,left_fitx, right_fitx)

plt.imshow(result)
plt.show()
```

The search can be illustrated as below:

![alt text][img_search_poly]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

The method to calculate the curvature radius is `get_curvature_radius`. This method accept coordinates in image plane pixels
and perform a conversion into real world value. The value used to convert for y axis is 30/720 meter per pixel and
3.7/830 meter per pixel on x axis. 30/720 is chose arbitrarily while 3.7/830 is based on average lane width from the image.

The following codes show how left, right and average curvatures are computed

```python
# the per pix can be modified like so
lane_finding.ym_per_pix=30/720
lane_finding.xm_per_pix=3.7/830

left_curverad, right_curverad = lane_finding.get_curvature_radius(ploty,left_fitx, right_fitx)
print(f"Left lane line curvature: {round(left_curverad,2)} m")
print(f"Right lane line curvature: {round(right_curverad,2)} m")

# The average from left and right
curverad = (left_curverad + right_curverad) / 2
```

For computing the vehicle position relative to center, the `get_vehicle_position` is the method that perform this operation.
The logic of the computation assumes the camera is positioned at the center of the image. The following snippet shows
the calculation.

```python
image_center = imshape[1] / 2
# vehicle position is at the bottom where x at max index
car_position = (left_fitx[-1] + right_fitx[-1]) / 2
return (image_center - car_position) * self.xm_per_pix
```

The `get_vehicle_position` can be invoked as such:

```python
pos = lane_finding.get_vehicle_position(image.shape,left_fitx, right_fitx)
print(f"Vehicle position with respect to center : {round(pos,2)} m")
```


#### 6. Plot lines back into the original image

The LaneFinding class provides this functionality inside the `unwarp` method.

```python
def unwarp(self, original_img, warped_binary, M_inv, left_fitx, right_fitx, ploty):
```

The lines are filled with green color first and wrap with matrix `M_inv` that returned from the perspective transform.
The following image illustrates the steps.


![alt text][img_unwarp]

#### 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The method `add_num_estimation` provides the implementation to draw the metrices into the final image.

```python
labelled_img = lane_finding.add_num_estimation(result,curverad, pos)
```

![alt text][img_final]

#### 8. Test Images

All the previous stages are tested on the test images and saved into `output_images` with prefixed with the stage involved.

The full codes that generate the following image can be found in [`camera calibration.ipynb`](<camera calibration.ipynb>) below the header `Run the Pipeline on test images`

![alt text][img_test_images]

---

### Pipeline (video)

#### 1. Project 

Here's a [link to my video result](./project_video_output.mp4)

[![video](https://img.youtube.com/vi/dg7BeNz5qtA/0.jpg)](https://www.youtube.com/watch?v=dg7BeNz5qtA)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The issue with my current implementation is it depends on hard coded trapezoidal points for perspective transformation in which doesn't hold in challenge video and even
can be noticed in project video. The following image shows the shape of anomalous filled output trapezoidal.

![alt text][img_anomaly_1]

The reason is the hard coded `src` doesn't fit correctly into the lane-line.

![alt text][img_anomaly_2]

The potential solution that I can think of is to find the points dynamically such as using the histogram method to locate the potential
points in left and right x coordinates.