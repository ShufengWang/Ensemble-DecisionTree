#include"common.h"



int main()
{

	char address[1024];
	
	ifstream fileOut;
	ofstream fileIn;

	srand(unsigned(time(NULL)));

	vector<cv::Point> tloc(1, cv::Point(0,0));

	sprintf(address, "%s", "models/slct_adaboost.txt");
	string modelName = string(address);
	Ptr<Boost> adaboost=Boost::load<Boost>(modelName);

	string comb_modelName = "models/comb_adaboost.txt";
	string comb_biasName = "models/comb_bias.txt";

	string DTreeName = "models/dtrees/dtree_";
	int treeNum = adaboost->getRoots().size();
	//for(int i=1; i<=treeNum; i++)
	//    seperate_adaboost(modelName, DTreeName, i);
	//adaboost->clear();

	// load dtrees
	vector<Ptr<Boost>> dtrees;
	for(int i=1; i<=treeNum; i++){
		sprintf(address, "%d%s", i, ".txt");
		string tFile = DTreeName + (string)address;
		Ptr<Boost> temp = Boost::load<Boost>(tFile);
		dtrees.push_back(temp);
	}







	/////////////////////////////////////////////
	// descriptor setting
	/////////////////////////////////////////////

	
	cv::Size windSize = cv::Size(60,60);
	cv::Size gridSize = cv::Size(6,6);
	int chanNum = 10;
	int tempNum = 1308;
	vector<cv::Mat> temps;
	for(int i=0; i<tempNum; i++){
		sprintf(address, "%s%d%s", "haarTemps/t_", i, ".txt");
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

    cv::Mat featFreq(featIdx.size(), 1, CV_32SC1, cv::Scalar(0));
	fileOut.open("featFreq.txt");
	for(int i=0; i<featFreq.rows; i++)
		fileOut>>featFreq.at<int>(i,0);
	fileOut.close();

	vector<vector<int>> slct_featIdx;
	for(int i=0; i<featIdx.size(); i++)
		if(featFreq.at<int>(i,0)>0)
			slct_featIdx.push_back(featIdx[i]);
	int FeaDim = slct_featIdx.size();
	cout<<"# features: "<<FeaDim<<endl;







	//////////////////////////////////////////////
	// load datasets
	//////////////////////////////////////////////


	string dataset = "F:/FastFood/";
	string targetFolder = "set1/";
	int targetNum = 3022;
	vector<string> tempFile;
	vector<vector<cv::Rect>> tempGT;
	load_annotation(dataset, targetFolder, targetNum, tempFile, tempGT);

	vector<string> trainFile;
	vector<vector<cv::Rect>> trainGT;
	vector<string> testFile;
	vector<vector<cv::Rect>> testGT;
	for(int i=0; i<tempFile.size(); i++){
		if(i<tempFile.size()/3-100)
			if(i%3==0){
				testFile.push_back(tempFile[i]);
				testGT.push_back(tempGT[i]);
		    }
		if(i>tempFile.size()/3+100)
			if(i%32==0){
				trainFile.push_back(tempFile[i]);
				trainGT.push_back(tempGT[i]);
		    }
	}






	////////////////////////////////////////////////
	//// prepare initial training data
	////////////////////////////////////////////////

	//int posNum = 0;
	//int negNum = 0;
	//for(int i=0; i<trainFile.size(); i++){
	//	string fileName = dataset + targetFolder + trainFile[i];
	//	cv::Mat img = cv::imread(fileName);
	//	int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
	//	cv::Mat img_border;
	//	cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);

	//	for(int j=0; j<trainGT[i].size(); j++){
	//		cv::Rect gt_rect(trainGT[i][j].x-0.33*trainGT[i][j].width+left, trainGT[i][j].y-0.33*trainGT[i][j].width+top, 
	//			1.66*trainGT[i][j].width, 1.66*trainGT[i][j].width);
	//		cv::Mat pos_patch = img_border(gt_rect);
	//		cv::resize(pos_patch, pos_patch, cv::Size(windSize.width, windSize.height));
	//		sprintf(address, "%s%d%s", "transf_pos/", posNum, ".jpg");
	//		imwrite(address, pos_patch);

	//		vector<float> temp_dscrpt;
	//		generate_feature(pos_patch, tloc, gridSize, temps, slct_featIdx, temp_dscrpt);
	//		cv::Mat m_dscrpt(temp_dscrpt);
	//		cv::transpose(m_dscrpt, m_dscrpt);

	//		vector<float> pos_dscrpt;
	//		for(int k=0; k<dtrees.size(); k++){
	//			cv::Mat pdct;
	//			dtrees[k]->predict(m_dscrpt, pdct, DTrees::PREDICT_SUM);
	//			pos_dscrpt.push_back(pdct.at<float>(0,0));
	//		}
	//		temp_dscrpt.clear();
	//		m_dscrpt.release();

	//		sprintf(address, "%s%d%s", "transf_pos/", posNum, ".txt");
	//		fileIn.open(address);
	//		for(int k=0; k<pos_dscrpt.size(); k++)
	//			fileIn<<pos_dscrpt[k]<<"\n";
	//		fileIn.close();
	//		pos_dscrpt.clear();
	//		posNum++;


	//		for(int d=0; d<10; d++){
	//			cv::Rect rand_rect(0,0,windSize.width,windSize.height);
	//			float ol = 0;
	//			do{ 
	//				rand_rect.x = random(0,img_border.cols-windSize.width);
	//				rand_rect.y = random(0,img_border.rows-windSize.height);
	//				ol = overlapRatio(gt_rect, rand_rect, 0);
	//			}while(ol>0.2);
	//			cv::Mat neg_patch = img_border(rand_rect);
	//			sprintf(address, "%s%d%s", "transf_neg/", negNum, ".jpg");
	//		    imwrite(address, neg_patch);

	//			vector<float> t_dscrpt;
	//			generate_feature(neg_patch, tloc, gridSize, temps, slct_featIdx, t_dscrpt);
	//			cv::Mat mt_dscrpt(t_dscrpt);
	//			cv::transpose(mt_dscrpt, mt_dscrpt);

	//			vector<float> neg_dscrpt;
	//			for(int k=0; k<dtrees.size(); k++){
	//				cv::Mat pdct;
	//				dtrees[k]->predict(mt_dscrpt, pdct, DTrees::PREDICT_SUM);
	//				neg_dscrpt.push_back(pdct.at<float>(0,0));
	//			}
	//			t_dscrpt.clear();
	//			mt_dscrpt.release();

	//			sprintf(address, "%s%d%s", "transf_neg/", negNum, ".txt");
	//			fileIn.open(address);
	//			for(int k=0; k<neg_dscrpt.size(); k++)
	//				fileIn<<neg_dscrpt[k]<<"\n";
	//			fileIn.close();
	//			neg_dscrpt.clear();
	//			negNum++;
	//		}
	//	}
	//}






	////////////////////////////////////////////////////////////
	//// bootstrapping
	////////////////////////////////////////////////////////////

	//int iteNum = 5;
	//for(int iId=0; iId<iteNum; iId++){
	//	cout<<"bootstrapping "<<iId+1<<"/"<<iteNum<<endl;

	//	cout<<"load training data ";
	//	cv::Mat trainData(posNum+negNum, dtrees.size(), CV_32FC1, cv::Scalar(0));
	//	for(int i=0; i<trainData.rows; i++){
	//		if(i<posNum)
	//			sprintf(address, "%s%d%s", "transf_pos/", i, ".txt");
	//		else
	//			sprintf(address, "%s%d%s", "transf_neg/", i-posNum, ".txt");
	//		fileOut.open(address);
	//		for(int j=0; j<trainData.cols; j++)
	//			fileOut>>trainData.at<float>(i,j);
	//		fileOut.close();
	//	}
	//	cout<<"pos "<<posNum<<" neg "<<negNum<<" ";

	//	sprintf(address, "%s", "trainData.txt");
	//	fileIn.open(address);
	//	for(int j=0; j<trainData.rows; j++){
	//		if(j<posNum)
	//			fileIn<<"+1 ";
	//		else
	//			fileIn<<"-1 ";
	//		for(int k=0; k<trainData.cols; k++)
	//			fileIn<<k+1<<":"<<trainData.at<float>(j,k)<<" ";
	//		fileIn<<"\n";
	//	}
	//	fileIn.close();


	//	// combine weak classifiers
	//	string sparseSVM = "svm.exe -t 0 -c 0.01 trainData.txt svm_model.txt";
	//	system(sparseSVM.c_str());
	//	trainData.release();

	//	cout<<"generate detector"<<endl;
	//	sprintf(address, "%s", "svm_model.txt");
	//	cv::Mat alpha;
	//	float rho;
	//	cv::Mat supVec;
	//	load_svm_model(address, dtrees.size(), alpha, rho, supVec);
	//	cv::Mat w;
	//	w = alpha*supVec;
	//	fileIn.open(comb_biasName);
	//	fileIn<<-rho<<"\n";
	//	fileIn.close();
	//	alpha.release();
	//	supVec.release();

	//	combine_dtrees(comb_modelName, DTreeName, w);
	//	w.release();


	//	if(iId<iteNum-1){
	//		double scaleFactor = 1.05;
	//		int staLevId = 2;
	//		int winStride = 8;
	//		float minThr = 0;
	//		int slctSampNum = 10;
	//		int newSampStaNo = negNum + 1;
	//		for(int i=0; i<trainFile.size(); i+=1){
	//			string fileName = dataset + targetFolder + trainFile[i];
	//	        cv::Mat img = cv::imread(fileName);
	//			vector<cv::Rect> found;
	//			vector<float> rspn;
	//			detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
	//				comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);

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
	//				for(int k=0; k<trainGT[i].size(); k++){
	//					cv::Rect gt_rect(trainGT[i][k].x-0.33*trainGT[i][k].width, trainGT[i][k].y-0.33*trainGT[i][k].width, 
	//		                        		1.66*trainGT[i][k].width, 1.66*trainGT[i][k].width);
	//					float ol = overlapRatio(gt_rect, found[id], 0);
	//					if(ol>max_ol)
	//						max_ol = ol;
	//				}
	//				if(max_ol<0.2){
	//					cv::Rect neg_rect(found[id].x+left, found[id].y+top, found[id].width, found[id].height);
	//					cv::Mat neg_patch = img_border(neg_rect);
	//					cv::resize(neg_patch, neg_patch, cv::Size(windSize.width,windSize.height));
	//					sprintf(address, "%s%d%s", "transf_neg/", newSampStaNo, ".jpg");
	//					imwrite(address, neg_patch);

	//					vector<float> temp_dscrpt;
	//					generate_feature(neg_patch, tloc, gridSize, temps, slct_featIdx, temp_dscrpt);
	//					cv::Mat m_dscrpt(temp_dscrpt);
	//					cv::transpose(m_dscrpt, m_dscrpt);

	//					vector<float> neg_dscrpt;
	//					for(int k=0; k<dtrees.size(); k++){
	//						cv::Mat pdct;
	//						dtrees[k]->predict(m_dscrpt, pdct, DTrees::PREDICT_SUM);
	//						neg_dscrpt.push_back(pdct.at<float>(0,0));
	//						pdct.release();
	//					}
	//					temp_dscrpt.clear();
	//					m_dscrpt.release();

	//					sprintf(address, "%s%d%s", "transf_neg/", newSampStaNo, ".txt");
	//					fileIn.open(address);
	//					for(int r=0; r<neg_dscrpt.size(); r++)
	//						fileIn<<neg_dscrpt[r]<<"\n";
	//					fileIn.close();
	//					neg_dscrpt.clear();
	//					newSampStaNo++;
	//				}
	//			}

	//			found.clear();
	//			rspn.clear();
	//		}

	//		int hardExampNum = newSampStaNo - 1 - negNum;
	//		negNum = newSampStaNo-1;
	//		cout<<"hardExmp "<<hardExampNum<<endl;
	//	}
	//	cout<<endl;
	//}









	///////////////////////////////////////
	// test model
	///////////////////////////////////////


	cout<<"test on full images"<<endl;
	double scaleFactor = 1.09;
	int winStride = 4;
	float minThr = -1;
	int staLevId = 3;
	float avgTime = 0;
	for(int i=0; i<testFile.size(); i++){
        cout<<testFile[i];
		string file_addr = dataset + targetFolder + testFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);
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
		found.clear();
		rspn.clear();
	}
	avgTime /= float(testFile.size());
	cout<<"average time consumption: "<<avgTime<<" s per image"<<endl;
	cout<<endl;




	cout<<"evaluation"<<endl;
	FILE *fp_det;
	fp_det = fopen("det.txt", "wb");
	for(float thr=1; thr>=-1; thr-=0.02){
		float totalObjNum = 0;
		float detectObjNum = 0;
		float falsePosNum = 0;

		string resultAddress = "res/";
		evaluate(resultAddress, testFile, testGT, true, 0.4, thr, totalObjNum, detectObjNum, falsePosNum);

		float missRate = 1 - detectObjNum/totalObjNum;
		float fppi = falsePosNum/float(testFile.size());
		cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
		fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	}
	fclose(fp_det);






	cout<<"display result"<<endl;
    double rspnThr = -0.15; 
	cv::namedWindow("detections");
	for(int i=0; i<testFile.size(); i++){
		string image_address = dataset + targetFolder + testFile[i];
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
			detection[j].y += int((12.0/60.0)*float(detection[j].height));
			detection[j].height = int((48.0/60.0)*float(detection[j].height));
			detection[j].x += int((12.0/60.0)*float(detection[j].width));
			detection[j].width = int((36.0/60.0)*float(detection[j].width));
		}

		for(int j=0; j<detection.size(); j++)
		    cv::rectangle(Im, cv::Point(detection[j].x, detection[j].y), cv::Point(detection[j].x+detection[j].width, 
			    detection[j].y+detection[j].height), CV_RGB(0,0,255), 2);

		for(int j=0; j<testGT[i].size(); j++)
			cv::rectangle(Im, cv::Point(testGT[i][j].x, testGT[i][j].y),
			    cv::Point(testGT[i][j].x+testGT[i][j].width,
				testGT[i][j].y+1.33*testGT[i][j].width), CV_RGB(255,0,0), 2);
	    
		cv::imshow("detections", Im);
		cv::waitKey();
	}
	cv::destroyWindow("detections");




}