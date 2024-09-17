/*
  Shikha Tiwari
  CS 5330 S24
  tasks.cpp - contains all the task required for object recognition like thresholding,cleanup, segmenetation.
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "csv_utils.h"
#include "tasks.h"
using namespace std;
using namespace cv;


//                               STEP-1     THRESHOLDING

// ***************************** Binary Thresholding ****************************************

int createBinaryThresholdImage(cv::Mat &src,cv::Mat &threshImage, int threshold, int max,const std::string& adaptiveMethod){

cv:: Mat greyImage,blurImage;
//Pre-Processing image using blur 
cv::blur(src, blurImage, cv::Size(5, 5));
cv::cvtColor( blurImage, greyImage, cv::COLOR_BGR2GRAY);
threshImage.create(greyImage.size(), greyImage.type());

for(int i=0;i<greyImage.rows;i++){//rows
    for(int j=0;j<greyImage.cols;j++){//cols
        uchar pixel = greyImage.at<uchar>(i, j);
        if(adaptiveMethod == "THRESH_BINARY_INV"){
            if(pixel > threshold)
                threshImage.at<uchar>(i, j) = 0;
            else
                threshImage.at<uchar>(i, j) = max;
        }
        else if(adaptiveMethod == "THRESH_BINARY"){
            if(pixel > threshold)
                threshImage.at<uchar>(i, j) = max;
            else
                threshImage.at<uchar>(i, j) = 0;
        }
    }
}    
return (0);
}

// ***************************** Adaptive Thresholding ****************************************


int createAdaptiveThresholdImage(cv::Mat &src,cv::Mat &threshImage, int max, const std::string& adaptiveMethod,const std::string& thresholdType,int C){
 
cv:: Mat greyImage,blurImage,dst;
//Pre-Processing image using blur and vonverting it to greyscale
cv::blur(src, blurImage, cv::Size(5, 5));
cv::cvtColor( blurImage, greyImage, cv::COLOR_BGR2GRAY);
threshImage.create(greyImage.size(), greyImage.type());
dst.create(greyImage.size(), greyImage.type());

int blockSize = 3;
for(int i=0;i<greyImage.rows;i++){//rows
    for(int j=0;j<greyImage.cols;j++){//cols
        uchar pixel = greyImage.at<uchar>(i, j);
        int sum=0;
        int count =0;
        for (int r = std::max(0, i - blockSize / 2); r <= std::min(greyImage.rows - 1, i + blockSize / 2); ++r) {
                for (int c = std::max(0, j - blockSize / 2); c <= std::min(greyImage.cols - 1, j + blockSize / 2); ++c) {
                    sum += greyImage.at<uchar>(r, c);
                    ++count;
                }
            }
        int localMean = static_cast<int>(sum / count);
        int localThreshold = localMean-C;
        if (adaptiveMethod == "THRESH_BINARY_INV") {
                if (greyImage.at<uchar>(i, j) > localThreshold)
                    threshImage.at<uchar>(i, j) = 0;
                else
                    threshImage.at<uchar>(i, j) = max;
            }
            else if (adaptiveMethod == "THRESH_BINARY") {
                if (greyImage.at<uchar>(i, j) > localThreshold)
                    threshImage.at<uchar>(i, j) = max;
                else
                    threshImage.at<uchar>(i, j) = 0;
    }
    
    }
}
return (0);
}

//                               STEP-2     CLEAN_UP IMAGE


// ***************************** EROSION ****************************************

void geterodedImage(cv::Mat& src,cv::Mat& erodedImage){

    int kernelSize = 8;
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F);
    erodedImage.create(src.size(), src.type());
    int ni,nj;
    for(int i=0;i<src.rows;i++){//rows
        for(int j=0;j<src.cols;j++){//cols
            bool erode= false;
            for(int k=0;k<kernel.rows;k++){//kernel rows
                for(int l=0;l<kernel.cols;l++){//kernel cols
                    ni = i - kernelSize/2 + k;
                    nj = j - kernelSize/2 + l;
                    if(ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                    if(src.at<uchar>(ni, nj)==0){
                        erode = true;
                        
                    }
                    }
                    if(erode)
                        break;
                }
            }
            erodedImage.at<uchar>(i, j) = erode ? 0 : 255;
        }
    }
    
}

// ***************************** DILATION ****************************************

void getdilatedImage(cv::Mat& src,cv::Mat& dilatedImage){

    int kernelSize = 15;
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F);
    dilatedImage.create(src.size(), src.type());
    int ni,nj;
    for(int i=0;i<src.rows;i++){//rows
        for(int j=0;j<src.cols;j++){//cols
            bool dilate= false;
            for(int k=0;k<kernel.rows;k++){//kernel rows
                for(int l=0;l<kernel.cols;l++){//kernel cols
                    ni = i - kernelSize/2 + k;
                    nj = j - kernelSize/2 + l;
                    if(ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                    if(src.at<uchar>(ni, nj)!=0){
                        dilate = true;
                        break;
                    }
                    }
                    if(dilate)
                        break;
                }
            }
            dilatedImage.at<uchar>(i, j) = dilate ? 255 : 0;
        }
    }
   
}

//                               STEP-3     SEGMENTATION

// ***************************** Segmentation using connectedComponentsWithStats opencv function ****************************************

std::vector<std::pair<int, int>> getConnectedComponents(cv::Mat& src,cv::Mat& labels,cv::Mat& stats,cv::Mat& centroids, cv::Mat color_img, int max_regions = -1,int connectivity = 4, float area_threshold = 50) {

    // constas 
    int num_labels;


    // applying connected component analysis
    num_labels = cv::connectedComponentsWithStats(src, labels, stats, centroids, connectivity = connectivity);
    std::vector<std::pair<int, int>> area_index_pairs;
    vector<cv::Vec3b> colors(num_labels+1);
    colors[0] = cv::Vec3b(0,0,0);

    // as we got the region map, let us remove the small areas
    for (int i = 1; i < num_labels ; i++) {
        
        //getting the area of each component and filtering them
        int area_of_component = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area_of_component < area_threshold) {
            //printf("Removing the %dth component as the area is too small\n",i);
            }

        //assigning a random color to remaining pixels
        else {
            area_index_pairs.push_back({area_of_component,i});
        }
    }

    //sorting 
    std::sort(area_index_pairs.rbegin(), area_index_pairs.rend());

        // Print the contents of the vector
    for (const auto& pair : area_index_pairs) {
        std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    }
    

    //limiting the segmentation to N regions
    if (max_regions > 0 && max_regions < area_index_pairs.size()) {
        area_index_pairs.resize(max_regions);
    }
    

    // Map to reorder labels and ignore small regions
    std::vector<int> label_map(num_labels, 0); // Initialize all to 0 (background)
    int new_label = 1;

    for (const auto& p : area_index_pairs) {
        label_map[p.second] = new_label++; // Assign new labels to valid regions
    }

    // Assigning a unique color to each valid region
    for (const auto& pair : area_index_pairs) {
        int label = pair.second;
        colors[label] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    // Reassign labels to keep only the largest N regions and ignore small ones
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int old_label = labels.at<int>(i, j);
            labels.at<int>(i, j) = label_map[old_label];
        }
    }

    // Display the labeled image with colored regions
    // cv::Mat color_img = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = labels.at<int>(i, j);
            if (label > 0) { // Ignore background
                cv::Vec3b color = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
                color_img.at<cv::Vec3b>(i, j) = colors[label];
            }
        }
    }

    // Display the regions
    cv::imshow("Labeled Image", color_img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return area_index_pairs;

}

//                               STEP-4     COMPUTING AND DISPLAY FEATURES

// ***************************** computeAndDisplayFeatures ****************************************


int computeAndDisplayFeatures(const cv::Mat& regionMap, int regionId, cv::Mat& displayOutput, std::vector<float>& feature_vector) {

    if (displayOutput.channels() == 1) {
        cv::cvtColor(displayOutput, displayOutput, cv::COLOR_GRAY2BGR);
    }

    // Find region mask
    cv::Mat regionMask = regionMap == regionId;

    // Calculate moments for the region
    cv::Moments moments = cv::moments(regionMask, true);

    // Ensure the region has non-zero area to avoid division by zero
    if (moments.m00 == 0) {
        return -1; // Region has zero area
    }

    // Central moments are used for calculating the axis of least central moment
    double mu20 = moments.mu20 / moments.m00;
    double mu02 = moments.mu02 / moments.m00;
    double mu11 = moments.mu11 / moments.m00;

    // Calculate the orientation (angle of least central moment)
    double angle = 0.5 * std::atan2(2 * mu11, mu20 - mu02);

    // Compute the centroid of the region
    double centerX = moments.m10 / moments.m00;
    double centerY = moments.m01 / moments.m00;
    cv::Point2f center(centerX, centerY);

    // Get oriented bounding box
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(regionMask, nonZeroPoints);

    // Convert nonZeroPoints to cv::Point2f for cv::minAreaRect
    std::vector<cv::Point2f> nonZeroPointsFloat;
    for (const auto& pt : nonZeroPoints) {
        nonZeroPointsFloat.push_back(cv::Point2f(pt.x, pt.y));
    }
    cv::RotatedRect rect = cv::minAreaRect(nonZeroPointsFloat);

    // Compute features
    float width = rect.size.width;
    float height = rect.size.height;
    float area = cv::contourArea(nonZeroPoints);
    float boundingBoxArea = width * height;
    float percentFilled = (area / boundingBoxArea) * 100;
    float ratio = height / width;

    // Add the values to feature_vector
    feature_vector.push_back(angle);
    feature_vector.push_back(ratio);

    // Determine the length of the orientation line
    double lineLength = std::max(width, height) / 2;

    // Calculate the end points of the orientation line based on the angle
    cv::Point2f endpoint1(centerX + lineLength * cos(angle), centerY + lineLength * sin(angle));
    cv::Point2f endpoint2(centerX - lineLength * cos(angle), centerY - lineLength * sin(angle));

    // Draw the orientation line on the displayOutput image
    cv::line(displayOutput, endpoint1, endpoint2, cv::Scalar(255, 0, 0), 2); // Blue line for orientation

    // Draw the oriented bounding box on the output image
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(displayOutput, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); // Green line for bounding box
    }

    return 1; // Successfully computed and displayed features
}


//                               STEP-5     COLLECTING TRAINING DATA AND STORING IT TO CSV

void collectTrainingData(cv::Mat& src, cv::Mat labels,std::vector<std::pair<int, int>>  area_index_pairs){

  
  char csv_path[256] = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/Project3/feature_vector.csv";

for (const auto& pair : area_index_pairs){
        cv::Mat displayOutput = src.clone();
        std::vector<float> feature_vector;
        feature_vector.clear();
        int num = computeAndDisplayFeatures(labels,pair.second,displayOutput,feature_vector);
        if (num == 1 )
        {
            // std::string result = "Features " + std::to_string(pair.second);
            cv::imshow("Features", displayOutput);
            char key = cv::waitKey(0);
            if (key == 'n' || key == 'N') 
            {
                char object_name[256];
                std::cout << "Enter the name of the object";
                std::cin >> object_name;
                int success = append_image_data_csv(csv_path, object_name, feature_vector, 0);
                if (success == 0) {
                    std::cout << "Success";
                }
                else {
                    std::cout << "Failure";
                    }
            }
        }
    }

}


//                               STEP-4     COMPUTING SCALED EUCLIDEAN DISTANCE
// can extend this function to other distance metrics by modifying if else conditions

std::vector<Distanceclass> computeDistance(std::vector<char*> srcimagefileNames, std::vector<std::vector<float>> srcfeaturesData, std::vector<float>& targetFeature, std::string distanceMethod,std::vector<float> featureSD){

    int n = srcfeaturesData.size();
    vector<Distanceclass> distances;

    if (distanceMethod == "scaledEuclidean"){
        for(int i=0;i<n;i++){
            vector<float> singleSrcFeature = srcfeaturesData[i];
            Distanceclass distance;
            distance.dist = scaledEuclidean(singleSrcFeature, targetFeature,featureSD);
            distance.filename = srcimagefileNames[i];
            distances.push_back(distance);
    }}

    return distances;
}

double scaledEuclidean(std::vector<float>& A, std::vector<float>& B,std::vector<float> featureSD) {
    double sum = 0.0;
    for (int i = 0; i < A.size(); ++i) {
        double diff = (A[i] - B[i]) / featureSD[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


void readFeatures(string filename, vector<char*>& fileNames, vector<vector<float>>& imageData) {

    char* fileName_char = new char[filename.length() + 1];
    strcpy(fileName_char, filename.c_str());
    read_image_data_csv(fileName_char, fileNames, imageData,0);
    
}

void calculateMean(vector<vector<float>> srcfeaturesData, std::vector<float>& featureMean) {
    
    // Get the number of features
    int numFeatures = srcfeaturesData[0].size();

    // Resize the featureMean vector to hold the means for each feature
    featureMean.resize(numFeatures);

    // Initialize the sums for each feature to zero
    for (int i = 0; i < numFeatures; ++i) {
        featureMean[i] = 0.0;
    }
    
    // Iterate over each data sample
    int numSamples = srcfeaturesData.size();
    for (int i = 0; i < numSamples; ++i) {
        // Iterate over each feature in the current data sample
        for (int j = 0; j < numFeatures; ++j) {
            // Add the value of the current feature to the sum for that feature
            featureMean[j] += srcfeaturesData[i][j];
        }
    }
    
    // Calculate the mean for each feature by dividing the sum by the number of samples
    for (int i = 0; i < numFeatures; ++i) {
        featureMean[i] /= numSamples;
    }


}

void calculateSD(vector<vector<float>> srcfeaturesData, std::vector<float>& featureMean,std::vector<float>& featureSD){

    // Get the number of features
    int numFeatures = srcfeaturesData[0].size();

    // Resize the featureSD vector to hold the standard deviations for each feature
    featureSD.resize(numFeatures);

    // Initialize the sums for each feature to zero
    for (int i = 0; i < numFeatures; ++i) {
        featureSD[i] = 0.0;
    }

    // Iterate over each data sample
    int numSamples = srcfeaturesData.size();
    for (int i = 0; i < numSamples; ++i) {
        // Iterate over each feature in the current data sample
        for (int j = 0; j < numFeatures; ++j) {
            // Add the squared difference between the value of the current feature and its mean
            featureSD[j] += (srcfeaturesData[i][j] - featureMean[j])*(srcfeaturesData[i][j] - featureMean[j]);
        }
    }

    // Calculate the standard deviation for each feature by dividing the sum of squared differences by the number of samples and taking the square root
    for (int i = 0; i < numFeatures; ++i) {
        featureSD[i] = sqrt(featureSD[i] / numSamples);
    }

}

/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug) {
  const int ORNet_size = 128;
  cv::Mat padImg;
  cv::Mat blob;
	
  cv::Mat roiImg = src( bbox );
  int top = bbox.height > 128 ? 10 : (128 - bbox.height)/2 + 10;
  int left = bbox.width > 128 ? 10 : (128 - bbox.width)/2 + 10;
  int bottom = top;
  int right = left;
	
  cv::copyMakeBorder( roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0  );
  cv::resize( padImg, padImg, cv::Size( 128, 128 ) );

  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) / 0.5, // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  128,   // subtract mean prior to scaling
			  false, // input is a single channel image
			  true,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!/fc1/Gemm" );

  if(debug) {
    cv::imshow( "pad image", padImg );
    std::cout << embedding << std::endl;
    cv::waitKey(0);
  }

  return(0);
}

//creating a normalized vector to get histogram intersection
std::vector<float> normalized_vector(std::vector<float> &vec) {

    //getting the sum of squared distance of vector 
    float sum = 0.0;
    for (auto& vec_elem : vec) {
        sum = sum + vec_elem*vec_elem;

    }
    float norm = std::sqrt(sum);

    std::vector<float> normalized_vector;

    for (int i = 0; i < vec.size(); i++){
        normalized_vector.push_back(vec[i]/norm);
    }

    return normalized_vector;

}


// gets cosine distance between two vectors
float cosine_distance(std::vector<float> &vec1, std::vector<float> &vec2) {

    if (vec1.size() != vec2.size()) {
        // got vectors of different size
        return -1.0f;
    }

    //normalizing the vectors
    std::vector<float> norm_vector_1;
    std::vector<float> norm_vector_2;

    norm_vector_1 = normalized_vector(vec1);
    norm_vector_2 = normalized_vector(vec2);

    float sum = 0.0;

    for (int i = 0; i < norm_vector_1.size(); i++) {
        sum = sum + norm_vector_1[i] * norm_vector_2[i];
    }

    return 1.0- sum;
}



// Function to find the index of an object in the objects vector
int findObjectIndex(const std::vector<std::string>& objects, const std::string& object) {
    auto it = std::find(objects.begin(), objects.end(), object);
    if (it != objects.end()) {
        return std::distance(objects.begin(), it);
    }
    return -1; // Object not found
}


//                      CONFUSION MATRIX 


void createConfusionMatrix(const std::string& trueLabel, const std::string& predictedLabel, const std::string& filename) {
    std::vector<std::string> objects = {"Coin", "Pen", "Phone", "Wallet", "Specs","Bowl","Cream","Tape","Watch","Key"};
    std::vector<std::vector<int>> confusionMatrix(objects.size(), std::vector<int>(objects.size(), 0));
    
    // Read existing confusion matrix from the file
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        // Skip the header line
        std::getline(file, line);
        int row = 0;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            int col = 0;
            std::getline(ss, item, ',');
            while (std::getline(ss, item, ',')) {
                
                confusionMatrix[row][col] = std::stoi(item);
                
                ++col;
            }
            ++row;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for reading.\n";
        return;
    }

    // Update confusion matrix with the new observation
    int trueIndex = findObjectIndex(objects, trueLabel);
    int predIndex = findObjectIndex(objects, predictedLabel);
    confusionMatrix[trueIndex][predIndex]++;

    /* Output the updated confusion matrix
    for (size_t i = 1; i < objects.size(); ++i) {
        std::cout << objects[i] << "\t";
        for (size_t j = 1; j < objects.size(); ++j) {
            std::cout << confusionMatrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }*/

    // Write updated confusion matrix back to the file
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << "*,Coin,Pen,Phone,Wallet,Specs,Bowl,Cream,Tape,Watch,Key\n";
        for (size_t i = 0; i < objects.size(); ++i) {
            outFile << objects[i] << ",";
            for (size_t j = 0; j < objects.size(); ++j) {
                outFile << confusionMatrix[i][j];
                if (j != objects.size() - 1) {
                    outFile << ",";
                }
            }
            outFile << "\n";
        }
        outFile.close();
        std::cout << "Confusion matrix updated and saved to " << filename << "\n";
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
}


// ******************** Add Key stroke Info on the video **********************************

void addKeystrokeInfo(cv::Mat& frame, const std::map<char, std::string>& keystrokeDescriptions) {
    int y = 20;
    for (const auto& [key, description] : keystrokeDescriptions) {
        putText(frame,  "Matching object name : " + description, Point(10, y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        y += 30;
    }
}
