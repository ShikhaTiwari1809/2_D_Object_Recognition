/*
  Bruce Maxwell
  CS 5330 F23
  Read and threshold an image, then run the grassfire transform
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

  cv::Mat src;    // input image
  cv::Mat thresh; // thresholded image
  cv::Mat dimg;   // distance image
  cv::Mat dst;    // result and view image
  char filename[256];

  // error checking
  if(argc < 2) {
    printf("usage: %s <image filename>\n", argv[0]);
    return(-1);
  }

  // grab the filename
  strcpy( filename, argv[1] );

  // read the file
  src = cv::imread( filename );

  if( src.data == NULL ) {
    printf("error: unable to read image %s\n", filename);
    return(-2);
  }
  cv::imshow("Image", src );
  // make the image grayscale
  cv::cvtColor( src, src, cv::COLOR_BGR2GRAY);
  cv::imshow("Gray Image", src );
  // threshold the image
  // threshold of 100
  // anything above threshold gets 255 **but** invert intensities
  // if pix < 100 then pix = 255:  white fg on black bg
  cv::threshold( src, thresh, 100, 255, cv::THRESH_BINARY_INV );
cv::imshow("thresh", thresh );
  // initialize distance transform image
  // be aware of the max distance this data type an hold (255)
  dimg = cv::Mat::zeros( thresh.size(), CV_8UC1 ); 

  // forward pass
  for(int i=1;i<thresh.rows;i++) {
    for(int j=1;j<thresh.cols;j++) {
      if( thresh.at<uchar>(i, j) > 0 ) { // foreground pixel
	int up = dimg.at<uchar>(i-1, j) + 1; // distance from up pixel
	int left = dimg.at<uchar>(i, j-1) + 1; // distance from left pixel
	dimg.at<uchar>(i,j) = up < left ? up : left; // assign minimum
      }
    }
  }
  cv::imshow("first pass", dimg );

  // second pass, reverse direction
  int maxval=0;
  for(int i=thresh.rows-2;i>=0;i--) {
    for(int j=thresh.cols-2;j>=0;j--) {
      if( dimg.at<uchar>(i, j) > 0 ) { // foreground pixel
	int cur = dimg.at<uchar>(i, j);
	int down = dimg.at<uchar>(i+1, j) + 1;
	int right = dimg.at<uchar>(i, j+1) + 1;
	cur = down < cur ? down : cur;
	cur = right < cur ? right : cur;
	dimg.at<uchar>(i, j) = cur;
	
	maxval = maxval < cur ? cur : maxval; // update biggest value in transform
      }
    }
  }

  // compute a visualization image
  dst = 255 * dimg / maxval;

  cv::imshow( "distance transform", dst );
  cv::moveWindow( "distance transform", 300, 100 );

  cv::waitKey(0);
  
  return(0);
}
