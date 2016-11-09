#include"common.h"


void generate_feature(cv::Mat img, vector<cv::Point> winLoc, cv::Size gridSize, vector<cv::Mat> temps, 
					 vector<vector<int>> featIdx, vector<float> &dscrpt)
{

	// compute channels
	vector<cv::Mat> channels;
	compute_channels(img, channels);



	// blur channel image
	for(int i=0; i<channels.size(); i++){
		blur(channels[i], channels[i], gridSize, cv::Point(-1,-1), 1);
	}


	// compute feature
	for(int i=0; i<winLoc.size(); i++){
		vector<float> temp_dscrpt;
		for(int j=0; j<featIdx.size(); j++){
			float posNum=0, posVal=0, negNum=0, negVal=0, value=0;
			int tId = featIdx[j][0];
			int cId = featIdx[j][1];
			for(int h=0; h<temps[tId].rows; h++){
				int x = winLoc[i].x + temps[tId].at<float>(h,0)*gridSize.width + 0.5*gridSize.width;
				int y = winLoc[i].y + temps[tId].at<float>(h,1)*gridSize.height + 0.5*gridSize.height;
				if(temps[tId].at<float>(h,2)>0){
					posVal += float(channels[cId].at<float>(y,x));
					posNum ++;
				}else{
					negVal += float(channels[cId].at<float>(y,x));
					negNum ++;
				}
			}
			value = posVal/(posNum+0.0000001) - negVal/(negNum+0.0000001);
			temp_dscrpt.push_back(value);
		}
		dscrpt.insert(dscrpt.end(), temp_dscrpt.begin(), temp_dscrpt.end());
	}


}

	



