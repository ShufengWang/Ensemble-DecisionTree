#include<iostream>
#include<fstream>
#include<stdio.h>
#include<cstdlib>
#include<io.h>
#include<math.h>
#include<malloc.h>
#include<time.h>
#include<vector>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include<opencv2/core/core.hpp>
#include<opencv.hpp>
#include<cv.h>
#include<opencv/ml.h>

using namespace std;
using namespace cv;
using namespace cv::ml;



void load_annotation(string datasetAddress, string testFolderName, int fileNum, vector<string> &fileName, 
					 vector<vector<cv::Rect>> &groundTruth);

void generate_haarTemp_1x1(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_1x2(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_1x3(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_1x4(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_2x1(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_2x2(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_2x3(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_2x4(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_3x1(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_3x2(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_3x3(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void generate_haarTemp_3x4(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps);

void load_templates(char *address, cv::Mat &haarTemp);

void compute_channels(cv::Mat img, vector<cv::Mat> &channels);

void generate_feature(cv::Mat img, vector<cv::Point> winLoc, cv::Size gridSize, vector<cv::Mat> temps, 
					 vector<vector<int>> featIdx, vector<float> &dscrpt);

double random(double start, double end);

float overlapRatio(cv::Rect &R1, cv::Rect &R2, int referTo);

void AdaBoost(cv::Mat trainData, cv::Mat response, int maxDepth, int weakCount, string modelName);

void seperate_adaboost(string& boost_model_file, string save_prefix, int dtree_id);

void combine_dtrees(string& boost_model_file, string prefix, cv::Mat weights);

void detect_adaboost(cv::Mat img, float scaleFactor, int staLevId, cv::Size winSize, int winStride, 
					 vector<cv::Rect> &found, vector<float> &rspn, string &modelAdd, vector<cv::Mat> temps, vector<vector<int>> featIdx,
					 cv::Size gridSize, float minThr);

void detect_comb_adaboost(cv::Mat img, float scaleFactor, int staLevId, cv::Size winSize, int winStride, 
					 vector<cv::Rect> &found, vector<float> &rspn, string &modelAdd, string &biasAdd, vector<cv::Mat> temps, vector<vector<int>> featIdx,
					 cv::Size gridSize, float minThr);

void pairwiseNonmaxSupp(vector<cv::Rect> &result, vector<float> &rspn, float overlapThr);

void removeCoveredRect(vector<cv::Rect> &result, vector<float> &rspn, float overlapThr);

void load_fgm_model(char *address, vector<int> &featIdx, vector<float> &coe);

void load_svm_model(char *address, int feaDim, cv::Mat &alpha, float &rho, cv::Mat &supVec);

void evaluate(string resultAddress, vector<string> fileName, vector<vector<cv::Rect>> groundTruth, 
	bool nonMaxSup, float matchThr, float rspnThr, float &totalObjNum, float &detectObjNum, float &falsePosNum);

void evaluate_whole(string resultAddress, vector<string> fileName, vector<vector<cv::Rect>> groundTruth, 
	bool nonMaxSup, float matchThr, float rspnThr, float &totalObjNum, float &detectObjNum, float &falsePosNum);
void evaluate_full(string resultAddress, vector<string> fileName, vector<vector<cv::Rect>> groundTruth, 
	bool nonMaxSup, float matchThr, float rspnThr, float &totalObjNum, float &detectObjNum, float &falsePosNum);


// added by wsf
void parseResultFromCNN(string dataset, string targetFolder,string subset);
void parseAnnotation(string datasetAddress, string testFolderName, string outFolder, int fileNum);
void parseCandidates(string datasetAddress, string testFolderName, string outFolder, int fileNum, vector<string> tempFile, string targetSubset);
bool inWhichSubset(int i,int fileNum, string & dataset_name);
void imageToVideo(string dataset, string targetFolder, string subFolder, vector<string> targetFile);