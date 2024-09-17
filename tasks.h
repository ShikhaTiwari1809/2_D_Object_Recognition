/*
  Shikha Tiwari
  CS 5330 S24
  tasks.h
*/


#ifndef GAUSS_H
#define GAUSS_H
#include <opencv2/opencv.hpp>

using namespace std;
class Distanceclass {

public:
	char* filename;
	float dist;
};



int createBinaryThresholdImage(cv::Mat &src,cv::Mat &threshImage, int threshold, int max,const std::string& adaptiveMethod);
int createAdaptiveThresholdImage(cv::Mat &src,cv::Mat &threshImage, int max, const std::string& adaptiveMethod, const std::string& thresholdType,int C);
void geterodedImage(cv::Mat& src,cv::Mat& erodedImage);
void getdilatedImage(cv::Mat& src,cv::Mat& dilatedImage);
std::vector<std::pair<int, int>> getConnectedComponents(cv::Mat& src,cv::Mat& labels,cv::Mat& stats,cv::Mat& centroids, cv::Mat color_img, int max_regions,int connectivity, float area_threshold);
int computeAndDisplayFeatures(const cv::Mat& regionMap, int regionId, cv::Mat& displayOutput, std::vector<float>& feature_vector);
void collectTrainingData(cv::Mat& src,cv::Mat labels,std::vector<std::pair<int, int>>  area_index_pairs);
std::vector<Distanceclass> computeDistance(std::vector<char*> srcimagefileNames, std::vector<std::vector<float>> srcfeaturesData, std::vector<float>& targetFeature, std::string distanceMethod,std::vector<float> featureSD);
double scaledEuclidean(std::vector<float>& A, std::vector<float>& B,std::vector<float> featureSD);
void readFeatures(string filename, vector<char*>& fileNames, vector<vector<float>>& imageData);
void calculateMean(vector<vector<float>> srcfeaturesData, std::vector<float>& featureMean);
void calculateSD(vector<vector<float>> srcfeaturesData, std::vector<float>& featureMean,std::vector<float>& featureSD);
float cosine_distance(std::vector<float> &vec1, std::vector<float> &vec2);
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug);
int findObjectIndex(const std::vector<std::string>& objects, const std::string& object);
void createConfusionMatrix( const std::string& trueLabel, const std::string& predictedLabel,const std::string& filename);
//void createConfusionMatrix( const std::string& trueLabel, const std::string& predictedLabel);
void addKeystrokeInfo(cv::Mat& frame, const std::map<char, std::string>& keystrokeDescriptions);

#endif