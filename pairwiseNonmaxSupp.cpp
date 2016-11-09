#include"common.h"


void pairwiseNonmaxSupp(vector<cv::Rect> &result, vector<float> &rspn, float overlapThr)
{
	vector<bool> covered;
	covered.resize(result.size());
	fill(covered.begin(), covered.end(), false);

	if(result.size()>0){
		cv::Mat mat_rspn(rspn);
		cv::Mat sortIndex;
		cv::sortIdx(mat_rspn, sortIndex, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);
		for(int i=0; i<sortIndex.rows; i++)
			for(int j=i+1; j<sortIndex.rows; j++)
				if(covered[sortIndex.at<int>(j,0)] == false){
					float area_i, area_j, ratio;
					ratio = overlapRatio(result[sortIndex.at<int>(i,0)], result[sortIndex.at<int>(j,0)], 0);
					if(ratio>=overlapThr)
						covered[sortIndex.at<int>(j,0)] = true;
				}
	}

	vector<cv::Rect> newresult;
	vector<float> newrspn;
	for(int i=0; i<result.size(); i++){
		if(covered[i]==false){
			newresult.push_back(result[i]);
			newrspn.push_back(rspn[i]);
		}
	}

	result.clear();
	rspn.clear();
	result.insert(result.end(), newresult.begin(), newresult.end());
	rspn.insert(rspn.end(), newrspn.begin(), newrspn.end());
	newresult.clear();
	newrspn.clear();



}