#include"common.h"


void detect_adaboost(cv::Mat img, float scaleFactor, int staLevId, cv::Size winSize, int winStride, 
					 vector<cv::Rect> &found, vector<float> &rspn, string &modelAdd, vector<cv::Mat> temps, vector<vector<int>> featIdx,
					 cv::Size gridSize, float minThr)
{

	Ptr<Boost> tboost=Boost::load<Boost>(modelAdd);
	

	vector<double> levelScale;
	double scale = 1;
	for(int level=0; level<winSize.width; level++){
		if((cvRound(double(img.cols)/scale)<winSize.width)||(cvRound(double(img.rows)/scale)<winSize.height))
			break;
		if(level>=staLevId)
		    levelScale.push_back(scale);
		scale*=scaleFactor;
	}

	int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
	cv::Mat img_border;
	cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);


	vector<cv::Mat> pyramid_img(levelScale.size());
	for(int level=0; level<levelScale.size(); level++)
		resize(img_border, pyramid_img[level], cv::Size(cvRound(double(img_border.cols)/levelScale[level]), 
			cvRound(double(img_border.rows)/levelScale[level])));

	for(int level=0; level<pyramid_img.size(); level++){
		cv::copyMakeBorder(pyramid_img[level], pyramid_img[level], 0, 4-pyramid_img[level].rows%4, 0, 4-pyramid_img[level].cols%4, cv::BORDER_REPLICATE);
	}

	vector<vector<cv::Point>> total_loc(levelScale.size());
	for(int level=0; level<levelScale.size(); level++)
	    for(int i=0; i<pyramid_img[level].rows-winSize.height+1; i+=winStride)
	    	for(int j=0; j<pyramid_img[level].cols-winSize.width+1; j+=winStride)
			    total_loc[level].push_back(cv::Point(j,i));




	vector<cv::Point> p_loc;
	vector<int> p_level;
	for(int level=0; level<levelScale.size(); level++){
		vector<float> dscrpt;
		generate_feature(pyramid_img[level], total_loc[level], gridSize, temps, featIdx, dscrpt);
		cv::Mat mDscrpt = cv::Mat(dscrpt).reshape(0,total_loc[level].size());
		cv::Mat pdct;
		tboost->predict(mDscrpt, pdct, DTrees::PREDICT_SUM);
		for(int i=0; i<total_loc[level].size(); i++)
			if(pdct.at<float>(i,0)>minThr){
				p_loc.push_back(total_loc[level][i]);
				rspn.push_back(pdct.at<float>(i,0));
				p_level.push_back(level);
			}
	}
	


	for(int i=0; i<p_loc.size(); i++){
		cv::Rect t_rect(cvRound(p_loc[i].x*levelScale[p_level[i]])-left, cvRound(p_loc[i].y*levelScale[p_level[i]])-top, 
			cvRound(winSize.width*levelScale[p_level[i]]), cvRound(winSize.height*levelScale[p_level[i]]));
		found.push_back(t_rect);
	}



	tboost->clear();
}