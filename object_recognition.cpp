// /*
//   Shikha Tiwari
//   CS 5330 S24

//   object_recognition.cpp - 
// */

#include <cstdio>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "tasks.h"
#include "csv_utils.h"

using namespace std;

int main(int argc, char *argv[]) {

  cv::Mat src;    // input image
  cv::Mat thresh; // thresholded image
  cv::Mat thresh2;
  cv::Mat dimg;   // distance image
  cv::Mat dst;    // result and view image
  char filename[256];
  char purpose[256];
  char feature_type[256];
  char dnn_model[256];
string matchineobjname;
  // error checking
  if(argc < 4) {
    printf("usage: %s <image filename>\n", argv[0]);
    return(-1);
  }

  // grab the filename
  strcpy( filename, argv[1] );
  strcpy(purpose, argv[2]) ;
  strcpy(feature_type, argv[3]);

  if (argc > 4){
    strcpy(dnn_model, argv[4]);
  }

  // read the file
  src = cv::imread( filename );

  if( src.data == NULL ) {
    printf("error: unable to read image %s\n", filename);
    return(-2);
  }
  cv::imshow("Image", src );

  // threshold the image
  createBinaryThresholdImage(src,thresh,80,255,"THRESH_BINARY_INV");
  createAdaptiveThresholdImage(src,thresh2,255, "THRESH_BINARY_INV", "THRESH_BINARY_INV",1);
  cv::imshow("thresh", thresh );
  cv::imshow("thresh 2", thresh2 );

  //Cleanup image
  cv::Mat erodedImage;
  cv::Mat dilatedImage;

  getdilatedImage(thresh,dilatedImage);
  geterodedImage(dilatedImage,erodedImage);
  
  cv::imshow("Eroded Image", erodedImage); 
  cv::imshow("Dilated Image", dilatedImage); 

  if (strcmp(feature_type, "segmentation") == 0) {
    //Segmenation

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids; 
    cv::Mat color_img = cv::Mat::zeros(erodedImage.size(), CV_8UC3);
    std::vector<std::pair<int, int>>  area_index_pairs = getConnectedComponents(erodedImage, labels, stats,centroids,color_img,-1,4,200.0f);

    //Get Features and save to csv file

    if (strcmp(purpose, "training") == 0){
      collectTrainingData(erodedImage,labels,area_index_pairs);
    }
    //read target image features
    else {
      std::vector<float> featureTarget;
      for(const auto &value:area_index_pairs) {
          std::vector<float> singlefeatureTarget;
          cv:: Mat displayOutput;
          displayOutput = erodedImage.clone();
          int num = computeAndDisplayFeatures(labels, value.second,displayOutput, singlefeatureTarget);
          if (num == 1 ){
          // std::string result = "Features " + std::to_string(pair.second);
          cv::imshow("Features", displayOutput);
          char key = cv::waitKey(0);
          if (key == 's' || key == 'S') {
              featureTarget=singlefeatureTarget;
          }
          if (key == 'q' || key == 'Q') {
            continue;
          }

      }
      }
      
      for (const auto& value : featureTarget) {
          std::cout <<"Target feature : "<< value << std::endl;
          
      }
      
      // read csv file which have all the object features
      std::vector<char*> srcobjectNames;
      vector<vector<float>> srcfeaturesData;
      std::string fileName = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/Project3/feature_vector.csv";
      readFeatures(fileName, srcobjectNames, srcfeaturesData);
      
      //calculate mean and sd of all features
      std::vector<float> featureMean;
      std::vector<float> featureSD;

      calculateMean(srcfeaturesData,featureMean);
      calculateSD(srcfeaturesData,featureMean,featureSD);
      
      // compute distance and get distance vector

      vector<Distanceclass> distances  = computeDistance(srcobjectNames, srcfeaturesData, featureTarget,"scaledEuclidean",featureSD);


      //sort and return top image

      sort(distances.begin(), distances.end(), 
      [](Distanceclass const& a, Distanceclass const& b) {
        return a.dist < b.dist;
      });

      
      for (const auto& distance : distances) {
          
          cout << distance.filename << "Distance : " << distance.dist << endl;
           matchineobjname= std::string(distance.filename);
          cout<<"Matching object is : "<<matchineobjname;
          map<char, string> keystrokeDescriptions;
          keystrokeDescriptions['Matching_object_name'] = matchineobjname;
          addKeystrokeInfo(color_img, keystrokeDescriptions);
          cv::imshow("Labeled Image with matching object name", color_img);
          break;
      }
    }
  }

  else if (strcmp(feature_type, "deep_embedding") == 0) {

    char deep_csv[256] = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/Project3/deep_features.csv";
    char model_name[256] = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/Project3/or2d-normmodel-007.onnx";
 

    // read the network
    cv::dnn::Net net = cv::dnn::readNet( model_name );
    printf("Network read successfully\n");

    /// print the names of the layers
    // std::vector<cv::String> names = net.getLayerNames();

    // for(int i=0;i<names.size();i++) {
    //   printf("Layer %d: '%s'\n", i, names[i].c_str() );
    // }

    // read image and convert it to greyscale
    cv::Mat src = cv::imread( filename );
    cv::cvtColor( src, src, cv::COLOR_BGR2GRAY );

    // the getEmbedding function wants a rectangle that contains the object to be recognized
    cv::Rect bbox( 0, 0, src.cols, src.rows );

    // get the embedding
    cv::Mat embedding;
    int success = getEmbedding( src, embedding, bbox, net, 1 );  // change the 1 to a 0 to turn off debugging

    if (strcmp(purpose,"training") == 0) {
      if (success == 0) {

          char object_name[256];
          std::cout << "Enter the object name";
          std::cin >> object_name;
          std::vector<float> feature_vector;
          for (int i = 0; i < embedding.rows; ++i) {
              for (int j = 0; j < embedding.cols; ++j) {
                  feature_vector.push_back(embedding.at<float>(i, j));
              }
          }
          append_image_data_csv(deep_csv, object_name, feature_vector, 0);
      }
    }

    else {
      
      if (success == 0) {
          std::vector<float> feature_vector;
          for (int i = 0; i < embedding.rows; ++i) {
              for (int j = 0; j < embedding.cols; ++j) {
                  feature_vector.push_back(embedding.at<float>(i, j));
              }
          }
      

        std::vector<char *> database_files;
        std::vector<std::vector<float>> database_features;
        std::vector<std::pair<float, std::string>> file_distance;
        cout << "pu";
        if (feature_vector.size() != 0) {
            int result = read_image_data_csv(deep_csv, database_files, database_features, true);
            if (result == 0) {

                for (int i = 0;i < database_features.size(); i++) {
                    float distance = cosine_distance(feature_vector, database_features[i]);
                    file_distance.push_back({distance, database_files[i]});
                }
            }

        }
        std::sort(file_distance.begin(), file_distance.end());
        if (file_distance.empty()){
            printf("No images and their distances");
        }

        else  {
            for (int i = 0; i < 1; i++) {
                matchineobjname = file_distance[i].second.c_str();
                printf("image name %s Distance  %.4f\n", file_distance[i].second.c_str(), file_distance[i].first);
            }
        }
      }

    }

  }

  // Creating Confusion Matrix , can do for all the set of images 
  string cmfile = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/Project3/deep_confusion_matrix.csv";
  //createConfusionMatrix("Key",matchineobjname,cmfile);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return(0);
}








