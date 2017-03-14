# SDC-behavioral-cloning
The goal of this project is to train a convolutional nueral net to generate the correct steering angles form training data. Note that eventhough the car drives autonomously on test track, the only prediction the nueralnet is making is the 'steering angle'. It will be interesting to further develop this idea to include to generate other required parameters like throttle which may require  a combination of sensor data and machine learning.  

Key steps of this pipeline are:
* Collect drving data from the simulator in training mode
* Augment data set by including images from left and right cameras and flipping images
* Build a image processing pipeline involving cropping and normalizing
* Build a CNN in keras with generators to flip and crop the image on the fly 
* Train and validate the model
* Test the car by running it in autonomous mode 

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
###Code

Run the python notebook `behavioral_cloning.ipynb` to implement the Keras model and . Implementation consists of the following files located in the source directory

* behavioral_cloning.ipynb          -   Implements CNN model and processing pipeline   
* drive.py                          -   Generates the steering angle predictions in autonomous mode
* model.h5                          -   Saved CNN model
* out_videos                        -   Folder with car driving in autonomous mode at multiple speeds
* writeup.md                        -   You are reading it

Recordings of driving in autonomous mode are available in the folder `out_videos`. The simulator i downloaded had a speed set to 9mph. Uploaded are videos in autonomous mode at speeds 9mph and 15 mph.

The actual model is implemented in `behavioural cloning.ipynb`. This file also contains data augmenting and processing routines.

###Model Architecture and Training Strategy

The model architecture is similar to the architecture proposed [here by NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This architecture has been demonstrated to work in real world setting and seems to have generated reasonably good results. Hence this network was chosen as the starting point. However, since our track and lane conditions are much simpler, the depth of the network and the nodes at each layer were reduced. As described below, the final model consists of 4 convolutional layers with 3x3 convolution windows. Relu activation and 2x2 max pooling is applied after each conv. layer. Finally 3 FC layers with dropout are utilized to estimate the output of steering angle. Most of the parameters such as window sizes, learning rate were finalized based on empirical data. 

The augmented data set was split into training and validation sets. Training and validation losses were monitored to ensure that the model is not overfitting the data. To better generalize, the driving data that was collected was augmented to reduce the driving biases associated with the data set. Also, dropout was used in the dense layers toward the output. It was also observed that 10 epochs of training are sufficient to run the car reasonably well in autonomous mode. There is room for a lot more optimization both in terms of augmenting the data and the model which will be done in the future.

The car runs easily at the default speed setting of 9 mph in the model. It also runs well at 15mph without crossing either of the lane boundaries. While there is a bit of moving sideways between the lanes and during the edges, this was primarily due to how the data was captured. The original data was captured at the fastest speed and did not necessarily keep the car always centered. This is another optimization that can be done easily in the near future.

###Final Model Architecture

The final model architecture is located in the file `behavioural cloning.ipynb` and is shown below. It consists of 4 convolution layers followed by 3 FC layers. Each convolution layers is followed by a Relu activation layers and a max pooling layer. A lamda layer that takes the cropped images and 

```python
    #Keras model
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Convolution2D, core, convolutional, MaxPooling2D, Lambda, Flatten

    model = Sequential()

    #Original image (160, 320, 3).. With cropping (70,320)
    model.add(Lambda(lambda x : (x-127.5)/127.5, input_shape = (70,320,3)))

    #adds 16 3x3 filters on input and a 2x2 max pooling
    #output after conv. is 16@68x318, after pooling 16@34x159
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #output after conv. is 24@32x157, after pooling 24@16x78        
    model.add(Convolution2D(24,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
          
    #output after conv. is 32@14x76, after pooling 32@7x38          
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #output after conv. is 64@5x36, after pooling 64@2x18          
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
          
    #output is 64X2X18 = 2304          
    model.add(Flatten())
    model.add(Dense(300, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))

```
Here is a visulation of network and output from the model that shows the parameters in each layer. In total there are ~750K parameters that are trained. 

```python
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 70, 320, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 68, 318, 16)   448         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 68, 318, 16)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 34, 159, 16)   0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 157, 24)   3480        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 157, 24)   0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 78, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 14, 76, 32)    6944        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 14, 76, 32)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 38, 32)     0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 36, 64)     18496       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 5, 36, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 2, 18, 64)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2304)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 300)           691500      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 300)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           30100       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            1010        dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 751,989
Trainable params: 751,989
Non-trainable params: 0
____________________________________________________________________________________________________

```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text](./writeup_images/pipeline.png)


3. Creation of the Training Set & Training Process

Data was captured from the simulator in training mode and augmented. Total data set includes
1) three laps of center driving on the original track  2) two laps of driving in reverse and 3) one lap of recovery. Data collection is oen of the most important parts of this project. One of the experiments that was to capture the data while driving the car at the maximum speed which meant that the corners were not taken at the middle of the road, but closer to the edge like in the real world. This results in the car behaving very similarly in autonomous mode as well. The car comes close to the edges while taking a turn but stays within lanes. 

Here is an example image of center lane driving:

alt text

While there are a lot of driving straight, 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Then I repeated this process on track two in order to get more data points.

Data was augmented in two ways
1) Inculding images from both the left and right cameras in the data set. Steering angle correction was left and right cameras was kept at 0.2 degrees. This is shown below

```python
    camera_adjust_angle = 0.2

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)    
        for row in reader:
            steering_center = float(row[3])
            steering_left = steering_center + camera_adjust_angle
            steering_right = steering_center - camera_adjust_angle        
```

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

alt text alt text

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.



















Finally calibration matrix (mtx) and distortion coefficients (dst) are calculated using the `cv2.calibrateCamera()` function
```python
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```    

To remove distortion in an image, the function `cv2.undistort()` is applied with calibration matrix and distortion coefficients found above.
```python
    dst = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
```

Applying this on chessboard images, we get 

![Original Distorted Image](./writeup_images/camera_dist_correct.png)

We can clearly see distortion at the top of the left image corrected after applying `cv2.undistort()`

###Image Pipeline 
####1. Distortion correction

Applying the same distortion correction as above
![alt text](./writeup_images/dist_road.png)

####2. Binary thresholding using Gradient and Color tranforms 

A combination of color and gradient thresholds was used to generate the binary image. Four different thresholds were used to generate the thresholded binary image. 

* S-color tranform
* SobelX gradient
* Sobel gradient magnitude
* Sobel gradient direction

The following thresholds were narrowed based on experimentation.

| Transform               | Threshold     | 
|:-----------------------:|:-------------:| 
| S color                 | 170, 255      | 
| SobelX grad             | 20, 100       |
| Sobel gradmagnitude     | 20, 100       |
| Sobel graddirection     | 0.7, 1.3      |

The final thresholded image is obtained by combining the various transforms as shown below. The code for thresholding is implemented in the file `source/gen_process_image.py`

```python
    combined_binary[(s_binary == 1) | (sxbinary == 1) | ((smagbinary == 1) & (sdirbinary == 1))] = 1
```

The images below show the effect of thresholding. The top image shows SobelX gradient and Color transform apllied, whereas the bottom image shows the result with all four thresholds applied

![alt text](./writeup_images/gradient_threshold.png)

####3. Perspective transform

The thresholded image is then run through a Perspective tranform to generate a birds-eye view image. This is accomplished by the opencv functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`

```python 
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
```

This source and destination points taken for the perspective transform are shown below.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 100, 0        | 
| 735, 460      | 1180, 0       |
| 0, 720        | 100, 720      |
| 1280, 720     | 1180, 0       |

As expected the source and destination points we pick impact the tranformed image quite a bit. This is more pronounced when the images contain shadows. An interesting observation is that occasionally better perspective transform and lane detection are achieved when the source images were taken to the ends of the image (rather than to the ends of the lane). 

Shown below are a thresholded image before and after the perspective transform is applied 

![alt text](./writeup_images/perspective.png)


####4. Identifying lane-lines and polyfit

The next step is to identify lane lines from the perspective trasformed image. For most instances, thresolding coupled with perspective transform provide reasonably clean outlines of the lane pixels. A sliding window technique is then used to identify the lane pixels. 

This section is implemented in `gen_lanefit.py`

First, a histogram of ON pixels is run the bottom half of image. 

```python
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
```

Then the location high intensity areas on the left and right sections of image are identified to give a starting location for the sliding window. 

```python
    end_margin_px = 100

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[end_margin_px:midpoint]) + end_margin_px
    rightx_base = np.argmax(histogram[midpoint+end_margin_px:histogram.shape[0]-100]) + midpoint + end_margin_px
```

The sliding window is moved along the the image and for each iteration of the window non-zero pixels in x and y direction are idenitifed.

```python
     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
```

These good indices are appended to an array. At the end of each iteration, the mean of non-zero pixels is used to center the sliding windows of the next iteration. If there are not enough pixels, then the location of the window stays the same as before. 

```python
    if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

Once the sliding window is moved across the entire image, the non-zero x and y pixels are curve fitted using a 2nd order polynomial to detect lane lines  

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

Shown below is the curve fitted lane lines with sliding windows and histogram of pixels  

![alt text](./writeup_images/curvefit.png)

Even in the limited test video provided, there are interesting cases where the entire thresholding and lane detection pipeline fails. They fall primarily in two areas
* Frames where the ends of the image do not have any active(ON) pixels since the line is dotted. Due to the nature of polyfit, this almost always returns an erroroneus fit
* Frames with shadows which make the processed images extremely noisy making it harder to even detect lane lines resulting in gross failures

Error correction for both these cases are implemented as shown below
In both these cases, the result is manifested as the right dotted white line detected being too far off (to the left or right) from its actual location. Here we measure the average road width and compare if it changed significantly (more than 15%) and apply correction  

First we measure the average roadwidth and curvature of the road as shown below

```python
    curr_road_width = np.average(right_fitx - left_fitx)    
    lc_rad, rc_rad = calc_curv(left_fitx, right_fitx, ploty)
```

If the detected roadwidth changes significantly compared to the previous frame, it is ignored. If the leftlane is calculated with good precision, then the right lane is calculated by just adding the average roadwidth to the left lane

```python
    if ((curr_road_width < 0.85*avg_road_width) | (curr_road_width > 1.15*avg_road_width) | (rc_rad < 50)):
         curr_road_width = avg_road_width
         right_fitx = left_fitx + curr_road_width
    else:
         avg_road_width = curr_road_width
```

####5. Metrics - Radius of curvature & Offset from center

Radius of curvature and vehicle offset from center is calculated in the file `gen_stats_display.py`

 First, the lanes detected in pixels are converted to lanes in real world meters and curve fitted 
```python
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
```    

and then radii of curvature are calculated based on the formula below

```python 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix 
                                + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix 
                                + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

Offset from center is calculated based on the assumption that the camera is the center of the image. 
```python
    xm_per_pix = 3.7/700
    
    offset_px = (center - 0.5*(leftx[y_eval] + rightx[y_eval]))   
    offset = xm_per_pix * offset_px
```
    
####6. Pipeline output
All the functions for polyfill `filled_image()` and anotation `anotate_image()` are included in the file `gen_stats_display.py`

First the detected lane is mapped on the warped image using the function `cv2.fillPoly()` and it is then converted into original image space using inverse perspective transform `cv2.warpPerspective()`

```python
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0]))     
```

This entire pipeline is implemented in the file `gen_detection_pipeline.py`. Shown below is an image before and after passing through the pipeline

![alt text](./writeup_images/pipeline.png)
---

###Video Output

Here are links to the [video output](./output_video/adv_lane_track.mp4).

Another version is shown [here](./output_video/adv_lane_track1.mp4). The difference in both videos is mostly due to the areas selected for perspective transform and thresholds selected for color and gradient transforms. 

---

###Discussion and further work
This project is aN introduction to camera calibration, color and perspective transforms and curve fitting functions. However, it is not very robust and depends heavily on many factors going right. 

As you can see the pipeline is not robust in areas where the road has strong shadows and is wobbly. Also sections of the road with lighter color(concrete sections) combined with reflections of the sun make detecting lane especially the white dotted right lines much harder. There is already significant volume of academic research on shadow detection and elimination in images and this is an area that i would like understand and implement in the near future.
