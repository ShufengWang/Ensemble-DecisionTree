#include"common.h"



int main()
{

	char address[1024];
	
	ifstream fileOut;
	ofstream fileIn;

	srand(unsigned(time(NULL)));

	vector<cv::Point> tloc(1, cv::Point(0,0));

	sprintf(address, "%s", "models_whole/adaboost.txt");
	string modelName = string(address);

	sprintf(address, "%s", "models_whole/slct_adaboost.txt");
	string slct_modelName = string(address);

	




	//////////////////////////////////////////////
	// load datasets
	//////////////////////////////////////////////

	string dataSet = "F:/INRIA/";
	string posFolder = "pos/";
	int posFileNum = 614;
	vector<string> posFile;
	vector<vector<cv::Rect>> posGT;
	load_annotation(dataSet, posFolder, posFileNum, posFile, posGT);

	string negFolder = "neg/";
	int negFileNum = 1218;
	vector<string> negFile;
	vector<vector<cv::Rect>> negGT;
	load_annotation(dataSet, negFolder, negFileNum, negFile, negGT);

	string testFolder = "test/";
	int testFileNum = 288;
	vector<string> testFile;
	vector<vector<cv::Rect>> testGT;
	load_annotation(dataSet, testFolder, testFileNum, testFile, testGT);





	
	/////////////////////////////////////////////
	// descriptor setting
	/////////////////////////////////////////////

	
	cv::Size windSize = cv::Size(60,120);
	cv::Size gridSize = cv::Size(6,6);
	int chanNum = 10;
	int tempNum = 2798;
	vector<cv::Mat> temps;
	for(int i=0; i<tempNum; i++){
		sprintf(address, "%s%d%s", "haarTemps_whole/t_", i, ".txt");
		cv::Mat hTemp;
		load_templates(address, hTemp);
		temps.push_back(hTemp);
	}
	cout<<"# templates: "<<tempNum<<endl;

	vector<vector<int>> featIdx;
	for(int i=0; i<temps.size(); i++)
		for(int j=0; j<chanNum; j++){
			vector<int> t;
			t.push_back(i);
			t.push_back(j);
			featIdx.push_back(t);
		}
	int FeaDim = featIdx.size();;
	cout<<"# features: "<<FeaDim<<endl;









 //   ///////////////////////////////////////
	//// train detector
	/////////////////////////////////////////


	//cout<<"prepare initial training data"<<endl;
	//int posSampNum = 0; //2474
	//for(int i=0; i<posFile.size(); i++){
	//	string fileName = dataSet + posFolder + posFile[i];
	//	cv::Mat img = cv::imread(fileName);
	//	int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
	//	cv::Mat img_border;
	//	cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);

	//	for(int j=0; j<posGT[i].size(); j++){
	//		cv::Rect gt_rect(posGT[i][j].x+0.5*posGT[i][j].width-0.3125*posGT[i][j].height+left, posGT[i][j].y-0.125*posGT[i][j].height+top, 
	//			0.625*posGT[i][j].height, 1.25*posGT[i][j].height);
	//		cv::Mat pos_patch = img_border(gt_rect);
	//		cv::resize(pos_patch, pos_patch, windSize);

	//		for(int k=0; k<2; k++){
	//			if(k==1)
	//				cv::flip(pos_patch, pos_patch, 1);
	//			sprintf(address, "%s%d%s", "inria_adaboost_pos_whole/", posSampNum, ".jpg");
	//			imwrite(address, pos_patch);

	//			vector<float> pos_dscrpt;
	//			generate_feature(pos_patch, tloc, gridSize, temps, featIdx, pos_dscrpt);
	//			sprintf(address, "%s%d%s", "inria_adaboost_pos_whole/", posSampNum, ".txt");
	//			fileIn.open(address);
	//			for(int r=0; r<pos_dscrpt.size(); r++)
	//					fileIn<<pos_dscrpt[r]<<"\n";
	//			fileIn.close();
	//			posSampNum++;
	//		}
	//	}
	//}

	//int negSampNum = 0; //2436
	//for(int i=0; i<negFile.size(); i++){
	//	string fileName = dataSet + negFolder + negFile[i];
	//	cv::Mat img = cv::imread(fileName);
	//	int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
	//	cv::Mat img_border;
	//	cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);

	//	for(int k=0; k<2; k++){
	//		cv::Rect rand_rect(0,0,windSize.width ,windSize.height);
	//		rand_rect.x = random(0,img_border.cols-windSize.width);
	//		rand_rect.y = random(0,img_border.rows-windSize.height);
	//		cv::Mat neg_patch = img_border(rand_rect);
	//		sprintf(address, "%s%d%s", "inria_adaboost_neg_whole/", negSampNum, ".jpg");
	//		imwrite(address, neg_patch);

	//		vector<float> neg_dscrpt;
	//		generate_feature(neg_patch, tloc, gridSize, temps, featIdx, neg_dscrpt);
	//		sprintf(address, "%s%d%s", "inria_adaboost_neg_whole/", negSampNum, ".txt");
	//		fileIn.open(address);
	//		for(int r=0; r<neg_dscrpt.size(); r++)
	//				fileIn<<neg_dscrpt[r]<<"\n";
	//		fileIn.close();
	//		negSampNum++;
	//	}
	//}




	//// bootstrapping
	//int iteNum = 7;
	//for(int iId=0; iId<iteNum; iId++){
	//	cout<<"bootstrapping "<<iId+1<<"/"<<iteNum<<endl;

	//	cout<<"load training data ";
	//	cv::Mat trainData(posSampNum+negSampNum, FeaDim, CV_32FC1, cv::Scalar(0));
	//	cv::Mat response(posSampNum+negSampNum, 1, CV_32S, cv::Scalar(0));
	//	int posCount = 0, negCount = 0;
	//	for(int i=0; i<trainData.rows; i++){
	//		if(i%(int(trainData.rows/10))==0 && i>0)
	//			cout<<" *";
	//		if(i<posSampNum){
	//			sprintf(address, "%s%d%s", "inria_adaboost_pos_whole/", i, ".txt");
	//	    	response.at<int>(i,0) = 1;
	//	    	posCount++;
	//		}else{
	//			sprintf(address, "%s%d%s", "inria_adaboost_neg_whole/", i-posSampNum, ".txt");
	//			response.at<int>(i,0) = -1;
	//			negCount++;
	//		}
	//		fileOut.open(address);
	//		for(int j=0; j<trainData.cols; j++)
	//			fileOut>>trainData.at<float>(i,j);
	//		fileOut.close();
	//	}
	//	cout<<" pos "<<posCount<<" neg "<<negCount<<endl;

	//	cout<<"boosting  "<<modelName<<endl;
	//	AdaBoost(trainData, response, 3, 1024, modelName);
	//	




	//	// feature selection
	//	Ptr<Boost> model=Boost::load<Boost>(modelName);
	//	vector<Boost::Node> nodes = model->getNodes();
	//	vector<Boost::Split> splits = model->getSplits();
 //       cv::Mat featFreq(featIdx.size(), 1, CV_32SC1, cv::Scalar(0));
	//	for(int i=0; i<nodes.size(); i++)
	//		if(nodes[i].split>=0)
	//			featFreq.at<int>(splits[nodes[i].split].varIdx,0)++;
	//	fileIn.open("featFreq_whole.txt");
	//	for(int i=0; i<featFreq.rows; i++)
	//		fileIn<<featFreq.at<int>(i,0)<<"\n";
	//	fileIn.close();

	//	vector<vector<int>> slct_featIdx;
	//	for(int i=0; i<featIdx.size(); i++)
	//		if(featFreq.at<int>(i,0)>0)
	//			slct_featIdx.push_back(featIdx[i]);

	//	cv::Mat slct_trainData(trainData.rows, slct_featIdx.size(), CV_32FC1, cv::Scalar(0));
	//	int count = 0;
	//	for(int i=0; i<trainData.cols; i++)
	//		if(featFreq.at<int>(i,0)>0){
	//			slct_trainData.col(count) += trainData.col(i);
	//			count++;
	//		}
	//	trainData.release();
	//	model->clear();

	//	cout<<"boosting  "<<slct_modelName<<endl;
	//	AdaBoost(slct_trainData, response, 3, 1024, slct_modelName);
	//	slct_trainData.release();
	//	response.release();





	//	// display frequent templates
	//	cv::Mat tempFreq(temps.size(), 1, CV_32FC1, cv::Scalar(0));
	//	for(int i=0; i<nodes.size(); i++)
	//		if(nodes[i].split>=0){
	//		    int idx = (int)((float)splits[nodes[i].split].varIdx/(float)chanNum);
	//		    tempFreq.at<float>(idx,0)++;
	//		}

	//	int freqTempNum = 50;
	//	cv::Mat freqTemps(5*windSize.height, 10*windSize.width, CV_32F, cv::Scalar(0));
	//	cv::Mat sortIndex;
	//	cv::sortIdx(tempFreq, sortIndex, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);
	//	for(int i=0; i<50; i++){
	//		int id = sortIndex.at<int>(i,0);
	//		cv::Mat tempWin(windSize.height, windSize.width, CV_32F, cv::Scalar(0));
	//		for(int j=0; j<temps[id].rows; j++){
	//			if(temps[id].at<float>(j,2)>0)
	//				tempWin(cv::Rect(temps[id].at<float>(j,0)*gridSize.width, temps[id].at<float>(j,1)*gridSize.height, gridSize.width, gridSize.height))=1.0;
	//			if(temps[id].at<float>(j,2)<0)
	//				tempWin(cv::Rect(temps[id].at<float>(j,0)*gridSize.width, temps[id].at<float>(j,1)*gridSize.height, gridSize.width, gridSize.height))=0.3;
	//		}
	//		tempWin.copyTo(freqTemps(cv::Rect((i%10)*windSize.width, (i/10)*windSize.height, windSize.width, windSize.height)));
	//	}
	//	freqTemps.convertTo(freqTemps, CV_8U, 255.0, 0.0);
	//	cv::namedWindow("freqTemps_whole");
	//	cv::imshow("freqTemps_whole", freqTemps);
	//	cv::waitKey(50);
	//	cv::imwrite("freqTemps_whole.png", freqTemps);



	//	




	//	if(iId<iteNum-1){
	//		double scaleFactor = 1.09;
	//		int staLevId = 3;
	//		int winStride = 8;
	//		float minThr = 0;
	//		int slctSampNum = 5;
	//		int newSampStaNo = negSampNum + 1;
	//		for(int i=0; i<negFile.size(); i+=1){
	//			string fileName = dataSet + negFolder + negFile[i];
	//	        cv::Mat img = cv::imread(fileName);
	//			vector<cv::Rect> found;
	//			vector<float> rspn;
	//			detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
	//				slct_modelName, temps, slct_featIdx, gridSize, minThr);

	//			cout<<i<<" "<<found.size()<<endl;
	//			int top = (int)(0.25*img.rows), left = (int)(0.25*img.cols); 
	//			cv::Mat img_border;
	//			cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);
	//			for(int j=0; j<min(slctSampNum,(int)found.size()); j++){
	//				int id;
	//				if(slctSampNum<found.size()){
	//					id = random(0, found.size()-1);
	//				}else{
	//					id = j;
	//				}

	//				float max_ol = 0;
	//				for(int k=0; k<negGT[i].size(); k++){
	//					cv::Rect gt_rect(negGT[i][k].x+0.5*negGT[i][k].width-0.3125*negGT[i][k].height, negGT[i][k].y-0.125*negGT[i][k].height, 
	//			                            0.625*negGT[i][k].height, 1.25*negGT[i][k].height);
	//					float ol = overlapRatio(gt_rect, found[id], 0);
	//					if(ol>max_ol)
	//						max_ol = ol;
	//				}
	//				if(max_ol<0.2){
	//					cv::Rect neg_rect(found[id].x+left, found[id].y+top, found[id].width, found[id].height);
	//					cv::Mat neg_patch = img_border(neg_rect);
	//					cv::resize(neg_patch, neg_patch, windSize);
	//					sprintf(address, "%s%d%s", "inria_adaboost_neg_whole/", newSampStaNo, ".jpg");
	//					imwrite(address, neg_patch);

	//					vector<float> neg_dscrpt;
	//					generate_feature(neg_patch, tloc, gridSize, temps, featIdx, neg_dscrpt);
	//					sprintf(address, "%s%d%s", "inria_adaboost_neg_whole/", newSampStaNo, ".txt");
	//					fileIn.open(address);
	//					for(int r=0; r<neg_dscrpt.size(); r++)
	//						fileIn<<neg_dscrpt[r]<<"\n";
	//					fileIn.close();
	//					newSampStaNo++;
	//				}
	//			}
	//		}

	//		int hardExampNum = newSampStaNo - 1 - negSampNum;
	//		negSampNum = newSampStaNo-1;
	//		cout<<"hardExmp "<<hardExampNum<<endl;
	//	}
	//	cout<<endl;
	//}


















	///////////////////////////////////////
	// test model
	///////////////////////////////////////


    cv::Mat featFreq(featIdx.size(), 1, CV_32SC1, cv::Scalar(0));
	fileOut.open("featFreq_whole.txt");
	for(int i=0; i<featFreq.rows; i++)
		fileOut>>featFreq.at<int>(i,0);
	fileOut.close();

	vector<vector<int>> slct_featIdx;
	for(int i=0; i<featIdx.size(); i++)
		if(featFreq.at<int>(i,0)>0)
			slct_featIdx.push_back(featIdx[i]);

	cout<<"test on full images"<<endl;
	double scaleFactor = 1.09;
	int winStride = 8;
	float minThr = -1;
	int staLevId = 3;
	float avgTime = 0;
	for(int i=0; i<testFile.size(); i++){
        cout<<testFile[i];
		string file_addr = dataSet + testFolder + testFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					slct_modelName, temps, slct_featIdx, gridSize, minThr);
		double time_1 = clock();
		double delta_time = (time_1-time_0)/CLOCKS_PER_SEC;
		avgTime += delta_time;
		cout<<" takes "<<delta_time<<" s"<<endl;
		
		string sFileName = testFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_addr = "res/" + sFileName + "txt";
		FILE *temp_fp = fopen(result_addr.c_str(), "wb");
		for(int j=0; j<found.size(); j++)
			fprintf(temp_fp, "%d %d %d %d %f\n", int(found[j].x), int(found[j].y), int(found[j].width), int(found[j].height), rspn[j]);
		fclose(temp_fp);
		img.release();
	}
	avgTime /= float(testFile.size());
	cout<<"average time consumption: "<<avgTime<<" s per image"<<endl;
	cout<<endl;






	cout<<"display result"<<endl;
    double rspnThr = 5; 
	cv::namedWindow("detections");
	for(int i=0; i<testFile.size(); i++){
		string image_address = dataSet + testFolder + testFile[i];
		cv::Mat Im = cv::imread(image_address);
		string sFileName = testFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_address = "res/" + sFileName + "txt";

		vector<cv::Rect> detection;
	    vector<float> rspn;
	    fileOut.open(result_address.c_str());
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

    	pairwiseNonmaxSupp(detection, rspn, 0.6);
	    removeCoveredRect(detection, rspn, 0.8);

		for(int j=0; j<detection.size(); j++){
			detection[j].y += int((12.0/120.0)*float(detection[j].height));
			detection[j].height = int((96.0/120.0)*float(detection[j].height));
			detection[j].x += int((12.0/60.0)*float(detection[j].width));
			detection[j].width = int((36.0/60.0)*float(detection[j].width));
		}

		for(int j=0; j<detection.size(); j++)
		    cv::rectangle(Im, cv::Point(detection[j].x, detection[j].y), cv::Point(detection[j].x+detection[j].width, 
			    detection[j].y+detection[j].height), CV_RGB(0,0,255), 2);

		//for(int j=0; j<testGT[i].size(); j++)
		//	cv::rectangle(Im, cv::Point(testGT[i][j].x, testGT[i][j].y),
		//	    cv::Point(testGT[i][j].x+testGT[i][j].width,
		//		testGT[i][j].y+testGT[i][j].width), CV_RGB(255,0,0), 2);
	    
		cv::imshow("detections", Im);
		cv::waitKey();
	}
	cv::destroyWindow("detections");






	return 0;
}