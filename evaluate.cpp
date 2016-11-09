#include"common.h"



void evaluate(string resultAddress, vector<string> fileName, vector<vector<cv::Rect>> groundTruth, 
	bool nonMaxSup, float matchThr, float rspnThr, float &totalObjNum, float &detectObjNum, float &falsePosNum)
{

	ifstream fileOut;

	for(int i=0; i<fileName.size(); i++){
		string sFileName = fileName[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string file_address = resultAddress + sFileName + "txt";

		// load results
		vector<cv::Rect> detection;
		vector<float> rspn;
		fileOut.open(file_address.c_str());
		while(!fileOut.eof()){
			cv::Rect temp_rect;
			float temp_rspn;
			fileOut>>temp_rect.x;
			fileOut>>temp_rect.y;
			fileOut>>temp_rect.width;
			fileOut>>temp_rect.height;
			fileOut>>temp_rspn;
			if(temp_rspn>rspnThr){
				detection.push_back(temp_rect);
				rspn.push_back(temp_rspn);
			}
		}
		fileOut.close();

		
		if(nonMaxSup==true){
	    	pairwiseNonmaxSupp(detection, rspn, 0.6);
		    removeCoveredRect(detection, rspn, 0.8);
		}

		

		// evaluate results
		totalObjNum += groundTruth[i].size();
		for(int j=0; j<groundTruth[i].size(); j++){
			if(detection.size()>0){
				cv::Rect gt;
				gt.x=groundTruth[i][j].x-0.33*groundTruth[i][j].width;
				gt.y=groundTruth[i][j].y-0.33*groundTruth[i][j].width;
				gt.width=1.66*groundTruth[i][j].width;
				gt.height=1.66*groundTruth[i][j].width;

				float maxVal = overlapRatio(gt, detection[0], 0);
				int maxIdx = 0;
				for(int k=1; k<detection.size(); k++){
			    	float ratio = overlapRatio(gt, detection[k], 0);
					if(ratio>maxVal){
						maxVal = ratio;
						maxIdx = k;
					} 
				}
				float matchRatio = overlapRatio(gt, detection[maxIdx], 0);
				if(matchRatio>=matchThr){
					detectObjNum++;
					detection.erase(detection.begin()+maxIdx);
				}
			}
		}
		falsePosNum += float(detection.size());
		detection.clear();
		rspn.clear();

	}

}