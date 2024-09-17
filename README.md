# 2_D_Object_Recognition
2 D Object Recognition using Open_CV in C++

## Instructions for running the code
- The code runs from command line
- Structure of the code  ./executable_file "image_path" "<training or inferece> " "<algorithm to use>" "<deep network path if using it>"

# Steps in 2 D object recognition
## Threshold the input video
- Command to get thresholded image
- The thresholded image will can be viewed in "thresh" window ./executable_file "image_path" "training" "segmentation" 

## Clean up the binary image
- Command to get the cleaned up image
- The eroded image can be viewed in "Eroded Image" window and dilated image can be viewed in "Dilated Image" ./executable_file "image_path" "training" "segmentation" 

## Segment the image into regions
- Command the segment the image
- Segmentation for the image can be viewed in "Labeled Image" window ./executable_file "image_path" "training" "segmentation" 

## Compute features for each major region
Command to compute features for each major region
- It will display the regions one by one. To computer features for it press "n" and if you want to skip the region press "q"
- If "n" is pressed, the system will ask the user to enter the object name in command line. ./executable_file "image_path" "training" "segmentation" 

## Classify new images
To classify images, run the following command
- The "Features" window will display each region one by one and if that region is the desired region to classify press "s" else press "q" ./executable_file "image_path" "inference" "segmentation" 

## Evaluate the performance of your system
- Confusion matrix is created dynamically for the system

Make sure to give the correct path and you can see confusion matrix in csv file being updated automatically

## Implement a second classification method
We implemented deep learning classifier

- Command to perform training with deep learning classifier ./executable_file "image_path" "training" "deep_embedding" "<model_path>" 

- Command to perform inference with deep learning classifier ./executable_file "image_path" "inference" "deep_embedding" "<model_path>" 

## Instructions for running the extensions
## Adaptive thresholding
Command to run adaptive thresholding
- The adpative thresholded image can be viewed in "thresh 2 window" ./executable_file "image_path" "training" "segmentation" 
## Training 10 Objects
We have trained the 10 objects for the system to recognise.
They can be viewed in feature_vector.csv file. Allowing the user to train multiple objects from just a single image. The system was built in such a way that it can train multiple objects from a single image
- Software will display the regions one by one. To computer features for it press "n" and if you want to skip the region press "q"
- If "n" is pressed, the system will ask the user to enter the object name in command line. ./executable_file "path_to_image_with_multiple_objects" "training" "segmentation" 
