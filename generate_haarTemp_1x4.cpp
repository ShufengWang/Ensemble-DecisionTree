#include"common.h"


void generate_haarTemp_1x4(cv::Size windSize, cv::Size gridSize, vector<cv::Mat> &temps)
{
	int ww = windSize.width;
	int wh = windSize.height;

	cv::Size blockSize(4*gridSize.width, 1*gridSize.height);
	int bw = blockSize.width;
	int bh = blockSize.height;

	int gw = gridSize.width;
	int gh = gridSize.height;

	int ncw = bw/gw;
	int nch = bh/gh;

	vector<cv::Point> blockLoc;
	for(int i=0; i<(windSize.height-blockSize.height)/gh+1; i++)
		for(int j=0; j<(windSize.width-blockSize.width)/gw+1; j++)
			blockLoc.push_back(cv::Point(j,i));

	for(int i=0; i<blockLoc.size(); i++){
		cv::Mat hTemp;
		for(int j=0; j<nch; j++)
			for(int k=0; k<ncw; k++){
				cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
				temp.at<float>(0,0) = float(blockLoc[i].x + k);
				temp.at<float>(0,1) = float(blockLoc[i].y + j);
				if(k==0)
				    temp.at<float>(0,2) = 1;
				else
					temp.at<float>(0,2) = -1;
				hTemp.push_back(temp);
				temp.release();
			}
			temps.push_back(hTemp);
	}

	//for(int i=0; i<blockLoc.size(); i++){
	//	cv::Mat hTemp;
	//	for(int j=0; j<nch; j++)
	//		for(int k=0; k<ncw; k++){
	//			cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
	//			temp.at<float>(0,0) = float(blockLoc[i].x + k);
	//			temp.at<float>(0,1) = float(blockLoc[i].y + j);
	//			if(k==1)
	//			    temp.at<float>(0,2) = 1;
	//			else
	//				temp.at<float>(0,2) = -1;
	//			hTemp.push_back(temp);
	//			temp.release();
	//		}
	//		temps.push_back(hTemp);
	//}

	//for(int i=0; i<blockLoc.size(); i++){
	//	cv::Mat hTemp;
	//	for(int j=0; j<nch; j++)
	//		for(int k=0; k<ncw; k++){
	//			cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
	//			temp.at<float>(0,0) = float(blockLoc[i].x + k);
	//			temp.at<float>(0,1) = float(blockLoc[i].y + j);
	//			if(k==2)
	//			    temp.at<float>(0,2) = 1;
	//			else
	//				temp.at<float>(0,2) = -1;
	//			hTemp.push_back(temp);
	//			temp.release();
	//		}
	//		temps.push_back(hTemp);
	//}

	for(int i=0; i<blockLoc.size(); i++){
		cv::Mat hTemp;
		for(int j=0; j<nch; j++)
			for(int k=0; k<ncw; k++){
				cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
				temp.at<float>(0,0) = float(blockLoc[i].x + k);
				temp.at<float>(0,1) = float(blockLoc[i].y + j);
				if(k==3)
				    temp.at<float>(0,2) = 1;
				else
					temp.at<float>(0,2) = -1;
				hTemp.push_back(temp);
				temp.release();
			}
			temps.push_back(hTemp);
	}

	for(int i=0; i<blockLoc.size(); i++){
		cv::Mat hTemp;
		for(int j=0; j<nch; j++)
			for(int k=0; k<ncw; k++){
				cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
				temp.at<float>(0,0) = float(blockLoc[i].x + k);
				temp.at<float>(0,1) = float(blockLoc[i].y + j);
				if(k==0 || k==1)
				    temp.at<float>(0,2) = 1;
				else
					temp.at<float>(0,2) = -1;
				hTemp.push_back(temp);
				temp.release();
			}
			temps.push_back(hTemp);
	}

	//for(int i=0; i<blockLoc.size(); i++){
	//	cv::Mat hTemp;
	//	for(int j=0; j<nch; j++)
	//		for(int k=0; k<ncw; k++){
	//			cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
	//			temp.at<float>(0,0) = float(blockLoc[i].x + k);
	//			temp.at<float>(0,1) = float(blockLoc[i].y + j);
	//			if(k==1 || k==2)
	//			    temp.at<float>(0,2) = 1;
	//			else
	//				temp.at<float>(0,2) = -1;
	//			hTemp.push_back(temp);
	//			temp.release();
	//		}
	//		temps.push_back(hTemp);
	//}

	//for(int i=0; i<blockLoc.size(); i++){
	//	cv::Mat hTemp;
	//	for(int j=0; j<nch; j++)
	//		for(int k=0; k<ncw; k++){
	//			cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
	//			temp.at<float>(0,0) = float(blockLoc[i].x + k);
	//			temp.at<float>(0,1) = float(blockLoc[i].y + j);
	//			if((j%2)!=(k%2))
	//			    temp.at<float>(0,2) = 1;
	//			else
	//				temp.at<float>(0,2) = -1;
	//			hTemp.push_back(temp);
	//			temp.release();
	//		}
	//		temps.push_back(hTemp);
	//}

}