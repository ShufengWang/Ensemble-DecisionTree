#include"common.h"



int main()
{

	char address[1024];
	
	ifstream fileOut;
	ofstream fileIn;

	srand(unsigned(time(NULL)));

	vector<cv::Point> tloc(1, cv::Point(0,0));

	sprintf(address, "%s", "models_whole/slct_adaboost.txt");
	string modelName = string(address);
	Ptr<Boost> adaboost=Boost::load<Boost>(modelName);

	string comb_modelName = "models_whole/comb_adaboost.txt";
	string comb_biasName = "models_whole/comb_bias.txt";

	string DTreeName = "models_whole/dtrees/dtree_";
	int treeNum = adaboost->getRoots().size();
	for(int i=1; i<=treeNum; i++)
	    seperate_adaboost(modelName, DTreeName, i);
	adaboost->clear();

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

    cv::Mat featFreq(featIdx.size(), 1, CV_32SC1, cv::Scalar(0));
	fileOut.open("featFreq_whole.txt");
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


	string dataset = "F:/";
	string targetFolder = "CUHK_Square_amplify/";
	int targetNum = 452;
	//string targetFolder = "MIT_Traffic_amplify/";
	//int targetNum = 520;
	vector<string> targetFile;
	vector<vector<cv::Rect>> targetGT;
	load_annotation(dataset, targetFolder, targetNum, targetFile, targetGT);

	vector<string> trainFile;
	vector<vector<cv::Rect>> trainGT;
	vector<string> testFile;
	vector<vector<cv::Rect>> testGT;
	vector<string> valFile;
	vector<vector<cv::Rect>> valGT;
	vector<string> unlabelFile;
	vector<vector<cv::Rect>> unlabelGT;

	for(int i=0; i<targetFile.size(); i++){
		if(i<352){
			if(i%20==0){
				trainFile.push_back(targetFile[i]);
				trainGT.push_back(targetGT[i]);
		    }
			else if(i%20==10){
				valFile.push_back(targetFile[i]);
				valGT.push_back(targetGT[i]);
			}
			else{
				unlabelFile.push_back(targetFile[i]);
				unlabelGT.push_back(targetGT[i]);
			}
		}
		else{
			if(i%1==0){
				testFile.push_back(targetFile[i]);
				testGT.push_back(targetGT[i]);
		    }
		}
	}


	////////////////////////////////////////////////
	//// prepare initial training data
	////////////////////////////////////////////////
	/*
	int posNum = 0;
	int negNum = 0;
	for(int i=0; i<trainFile.size(); i++){
		string fileName = dataset + targetFolder + trainFile[i];
		cv::Mat img = cv::imread(fileName);
		int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
		cv::Mat img_border;
		cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);

		for(int j=0; j<trainGT[i].size(); j++){
			cv::Rect gt_rect(trainGT[i][j].x+0.5*trainGT[i][j].width-0.3125*trainGT[i][j].height+left, trainGT[i][j].y-0.125*trainGT[i][j].height+top, 
				0.625*trainGT[i][j].height, 1.25*trainGT[i][j].height);
			cv::Mat pos_patch = img_border(gt_rect);
			cv::resize(pos_patch, pos_patch, windSize);
			sprintf(address, "%s%d%s", "transf_pos_whole/", posNum, ".jpg");
			imwrite(address, pos_patch);

			vector<float> temp_dscrpt;
			generate_feature(pos_patch, tloc, gridSize, temps, slct_featIdx, temp_dscrpt);
			cv::Mat m_dscrpt(temp_dscrpt);
			cv::transpose(m_dscrpt, m_dscrpt);

			vector<float> pos_dscrpt;
			for(int k=0; k<dtrees.size(); k++){
				cv::Mat pdct;
				dtrees[k]->predict(m_dscrpt, pdct, DTrees::PREDICT_SUM);
				pos_dscrpt.push_back(pdct.at<float>(0,0));
			}
			temp_dscrpt.clear();
			m_dscrpt.release();

			sprintf(address, "%s%d%s", "transf_pos_whole/", posNum, ".txt");
			fileIn.open(address);
			for(int k=0; k<pos_dscrpt.size(); k++)
				fileIn<<pos_dscrpt[k]<<"\n";
			fileIn.close();
			pos_dscrpt.clear();
			posNum++;


			for(int d=0; d<10; d++){
				cv::Rect rand_rect(0,0,windSize.width,windSize.height);
				float ol = 0;
				do{ 
					rand_rect.x = random(0,img_border.cols-windSize.width);
					rand_rect.y = random(0,img_border.rows-windSize.height);
					ol = overlapRatio(gt_rect, rand_rect, 0);
				}while(ol>0.2);
				cv::Mat neg_patch = img_border(rand_rect);
				sprintf(address, "%s%d%s", "transf_neg_whole/", negNum, ".jpg");
			    imwrite(address, neg_patch);

				vector<float> t_dscrpt;
				generate_feature(neg_patch, tloc, gridSize, temps, slct_featIdx, t_dscrpt);
				cv::Mat mt_dscrpt(t_dscrpt);
				cv::transpose(mt_dscrpt, mt_dscrpt);

				vector<float> neg_dscrpt;
				for(int k=0; k<dtrees.size(); k++){
					cv::Mat pdct;
					dtrees[k]->predict(mt_dscrpt, pdct, DTrees::PREDICT_SUM);
					neg_dscrpt.push_back(pdct.at<float>(0,0));
				}
				t_dscrpt.clear();
				mt_dscrpt.release();

				sprintf(address, "%s%d%s", "transf_neg_whole/", negNum, ".txt");
				fileIn.open(address);
				for(int k=0; k<neg_dscrpt.size(); k++)
					fileIn<<neg_dscrpt[k]<<"\n";
				fileIn.close();
				neg_dscrpt.clear();
				negNum++;
			}
		}
	}


	//////////////////////////////////////////////////////////
	// bootstrapping
	//////////////////////////////////////////////////////////

	int iteNum = 5;
	for(int iId=0; iId<iteNum; iId++){
		cout<<"bootstrapping "<<iId+1<<"/"<<iteNum<<endl;

		cout<<"load training data ";
		cv::Mat trainData(posNum+negNum, dtrees.size(), CV_32FC1, cv::Scalar(0));
		for(int i=0; i<trainData.rows; i++){
			if(i<posNum)
				sprintf(address, "%s%d%s", "transf_pos_whole/", i, ".txt");
			else
				sprintf(address, "%s%d%s", "transf_neg_whole/", i-posNum, ".txt");
			fileOut.open(address);
			for(int j=0; j<trainData.cols; j++)
				fileOut>>trainData.at<float>(i,j);
			fileOut.close();
		}
		cout<<"pos "<<posNum<<" neg "<<negNum<<" ";

		sprintf(address, "%s", "trainData_whole.txt");
		fileIn.open(address);
		for(int j=0; j<trainData.rows; j++){
			if(j<posNum)
				fileIn<<"+1 ";
			else
				fileIn<<"-1 ";
			for(int k=0; k<trainData.cols; k++)
				fileIn<<k+1<<":"<<trainData.at<float>(j,k)<<" ";
			fileIn<<"\n";
		}
		fileIn.close();


		// combine weak classifiers
		string sparseSVM = "svm.exe -t 0 -c 0.01 trainData_whole.txt svm_model_whole.txt";
		system(sparseSVM.c_str());
		trainData.release();

		cout<<"generate detector"<<endl;
		sprintf(address, "%s", "svm_model_whole.txt");
		cv::Mat alpha;
		float rho;
		cv::Mat supVec;
		load_svm_model(address, dtrees.size(), alpha, rho, supVec);
		cv::Mat w;
		w = alpha*supVec;
		fileIn.open(comb_biasName);
		fileIn<<-rho<<"\n";
		fileIn.close();
		alpha.release();
		supVec.release();

		combine_dtrees(comb_modelName, DTreeName, w);
		w.release();


		if(iId<iteNum-1){
			double scaleFactor = 1.09;
			int staLevId = 3;
			int winStride = 8;
			float minThr = 0;
			int slctSampNum = 10;
			int newSampStaNo = negNum;
			for(int i=0; i<trainFile.size(); i+=1){
				string fileName = dataset + targetFolder + trainFile[i];
		        cv::Mat img = cv::imread(fileName);
				vector<cv::Rect> found;
				vector<float> rspn;
				detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);

				cout<<i<<" "<<found.size()<<endl;
				int top = (int)(0.21*img.rows), left = (int)(0.21*img.cols); 
				cv::Mat img_border;
				cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);
				for(int j=0; j<min(slctSampNum,(int)found.size()); j++){
					int id;
					if(slctSampNum<found.size()){
						id = random(0, found.size()-1);
					}else{
						id = j;
					}

					float max_ol = 0;
					for(int k=0; k<trainGT[i].size(); k++){
						cv::Rect gt_rect(trainGT[i][k].x+0.5*trainGT[i][k].width-0.3125*trainGT[i][k].height, trainGT[i][k].y-0.125*trainGT[i][k].height, 
				                            0.625*trainGT[i][k].height, 1.25*trainGT[i][k].height);
						float ol = overlapRatio(gt_rect, found[id], 0);
						if(ol>max_ol)
							max_ol = ol;
					}
					if(max_ol<0.2){
						cv::Rect neg_rect(found[id].x+left, found[id].y+top, found[id].width, found[id].height);
						cv::Mat neg_patch = img_border(neg_rect);
						cv::resize(neg_patch, neg_patch, windSize);
						sprintf(address, "%s%d%s", "transf_neg_whole/", newSampStaNo, ".jpg");
						imwrite(address, neg_patch);

						vector<float> temp_dscrpt;
						generate_feature(neg_patch, tloc, gridSize, temps, slct_featIdx, temp_dscrpt);
						cv::Mat m_dscrpt(temp_dscrpt);
						cv::transpose(m_dscrpt, m_dscrpt);

						vector<float> neg_dscrpt;
						for(int k=0; k<dtrees.size(); k++){
							cv::Mat pdct;
							dtrees[k]->predict(m_dscrpt, pdct, DTrees::PREDICT_SUM);
							neg_dscrpt.push_back(pdct.at<float>(0,0));
							pdct.release();
						}
						temp_dscrpt.clear();
						m_dscrpt.release();

						sprintf(address, "%s%d%s", "transf_neg_whole/", newSampStaNo, ".txt");
						fileIn.open(address);
						for(int r=0; r<neg_dscrpt.size(); r++)
							fileIn<<neg_dscrpt[r]<<"\n";
						fileIn.close();
						neg_dscrpt.clear();
						newSampStaNo++;
					}
				}

				found.clear();
				rspn.clear();
			}

			int hardExampNum = newSampStaNo - negNum;
			negNum = newSampStaNo;
			cout<<"hardExmp "<<hardExampNum<<endl;
		}
		cout<<endl;
	}

	*/






	
	///////////////////////////////////////
	// test model
	///////////////////////////////////////
	string subset = "test";

	cout<<"test on full images"<<endl;
	double scaleFactor = 1.05;
	int winStride = 8;
	float minThr = -1;
	int staLevId = 3;
	float avgTime = 0;
	FILE *fp_det;
	
	/*
	for(int i=0; i<testFile.size(); i++){
        cout<<testFile[i];
		string file_addr = dataset + targetFolder + testFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		//detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
		//			modelName, temps, slct_featIdx, gridSize, minThr);
		detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);
		double time_1 = clock();
		double delta_time = (time_1-time_0)/CLOCKS_PER_SEC;
		avgTime += delta_time;
		cout<<" takes "<<delta_time<<" s"<<endl;
		
		string sFileName = testFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_addr = "results/" + subset + "/" + sFileName + "txt";
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
	*/

	
	//parseResultFromCNN(dataset, targetFolder, subset);
	//double threshold_fppi_1;
	//cout<<"evaluation"<<endl;
	//
	//fp_det = fopen(("det/" + subset + "_det.txt").c_str(), "wb");
	//for(float thr=1; thr>=-1; thr-=0.02){
	//	float totalObjNum = 0;
	//	float detectObjNum = 0;
	//	float falsePosNum = 0;
	//
	//	string resultAddress = "res_cnn/" + subset + "_2/";
	//	evaluate_whole(resultAddress, testFile, testGT, true, 0.5, thr, totalObjNum, detectObjNum, falsePosNum);
	//	float missRate = 1 - detectObjNum/totalObjNum;
	//	float fppi = falsePosNum/float(testFile.size());
	//	if(fppi<1.05) threshold_fppi_1 = thr;
	//	cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
	//	fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	//}
	//fclose(fp_det);
	//
	//
	//
	//
	//
	//
	//cout<<"display result"<<endl;
 //   double rspnThr = -0.15;
	//rspnThr = threshold_fppi_1;
	//cv::namedWindow("detections");
	//for(int i=0; i<testFile.size(); i++){
	//	string image_address = dataset + targetFolder + testFile[i];
	//	cv::Mat Im = cv::imread(image_address);
	//	string sFileName = testFile[i];
	//	sFileName.erase(sFileName.end()-3, sFileName.end());
	//	string result_address = "res_cnn/" + subset + "_2/" + sFileName + "txt";

	//	vector<cv::Rect> detection;
	//    vector<float> rspn;
	//    fileOut.open(result_address.c_str());
	//    while(!fileOut.eof()){
	//	    cv::Rect temp_rect;
	//	    float temp_rspn;
	//	    fileOut>>temp_rect.x;
	//	    fileOut>>temp_rect.y;
	//		fileOut>>temp_rect.width;
	//		fileOut>>temp_rect.height;
	//	    fileOut>>temp_rspn;
	//		if(temp_rspn>rspnThr){
	//			detection.push_back(temp_rect);
	//			rspn.push_back(temp_rspn);
	//		}
	//	}
	//    fileOut.close();

 //   	pairwiseNonmaxSupp(detection, rspn, 0.6);
	//    removeCoveredRect(detection, rspn, 0.8);

	//	for(int j=0; j<detection.size(); j++){
	//		detection[j].y += int((12.0/120.0)*float(detection[j].height));
	//		detection[j].height = int((96.0/120.0)*float(detection[j].height));
	//		detection[j].x += int((12.0/60.0)*float(detection[j].width));
	//		detection[j].width = int((36.0/60.0)*float(detection[j].width));
	//	}

	//	for(int j=0; j<detection.size(); j++)
	//	    cv::rectangle(Im, cv::Point(detection[j].x, detection[j].y), cv::Point(detection[j].x+detection[j].width, 
	//		    detection[j].y+detection[j].height), CV_RGB(0,0,255), 2);

	//	
	//	//for(int j=0; j<testGT[i].size(); j++)
	//	//	cv::rectangle(Im, cv::Point(testGT[i][j].x, testGT[i][j].y),
	//	//	    cv::Point(testGT[i][j].x+testGT[i][j].width,
	//	//		testGT[i][j].y+testGT[i][j].height), CV_RGB(255,0,0), 2);
	// //   
	//	string output_addr = dataset + targetFolder + "ACCV/detection_results/" + testFile[i];
	//	cv::imwrite(output_addr, Im);
	//	cv::imshow("detections", Im);
	//	cv::waitKey();
	//}
	//cv::destroyWindow("detections");
	//
	//imageToVideo(dataset, targetFolder, "ACCV/detection_results/", testFile);
	

	/*
	///////////////////////////////////////
	// test on train subset
	///////////////////////////////////////
	subset = "train";

	cout<<"test on train images"<<endl;
	avgTime = 0;
	for(int i=0; i<trainFile.size(); i++){
        cout<<trainFile[i];
		string file_addr = dataset + targetFolder + trainFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		//detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
		//			modelName, temps, slct_featIdx, gridSize, minThr);
		detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);
		double time_1 = clock();
		double delta_time = (time_1-time_0)/CLOCKS_PER_SEC;
		avgTime += delta_time;
		cout<<" takes "<<delta_time<<" s"<<endl;
		
		string sFileName = trainFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_addr = "results/" + subset + "/" + sFileName + "txt";
		FILE *temp_fp = fopen(result_addr.c_str(), "wb");
		for(int j=0; j<found.size(); j++)
			fprintf(temp_fp, "%d %d %d %d %f\n", int(found[j].x), int(found[j].y), int(found[j].width), int(found[j].height), rspn[j]);
		fclose(temp_fp);
		img.release();
		found.clear();
		rspn.clear();
	}
	avgTime /= float(trainFile.size());
	cout<<"average time consumption: "<<avgTime<<" s per image"<<endl;
	cout<<endl;
	*/


	
	/*cout<<"evaluation"<<endl;
	
	fp_det = fopen(("det/" + subset + "_det.txt").c_str(), "wb");
	for(float thr=1; thr>=-1; thr-=0.02){
		float totalObjNum = 0;
		float detectObjNum = 0;
		float falsePosNum = 0;

		string resultAddress = "results/" + subset + "/";
		evaluate_whole(resultAddress, trainFile, trainGT, true, 0.5, thr, totalObjNum, detectObjNum, falsePosNum);

		float missRate = 1 - detectObjNum/totalObjNum;
		float fppi = falsePosNum/float(trainFile.size());
		cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
		fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	}
	fclose(fp_det);
	*/
	


	/*
	cout<<"display result"<<endl;
    double rspnThr = -0.15; 
	cv::namedWindow("detections");
	for(int i=0; i<trainFile.size(); i++){
		string image_address = dataset + targetFolder + trainFile[i];
		cv::Mat Im = cv::imread(image_address);
		string sFileName = trainFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_address = "res/" + subset + "/" + sFileName + "txt";

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

		for(int j=0; j<trainGT[i].size(); j++)
			cv::rectangle(Im, cv::Point(trainGT[i][j].x, trainGT[i][j].y),
			    cv::Point(trainGT[i][j].x+trainGT[i][j].width,
				trainGT[i][j].y+trainGT[i][j].height), CV_RGB(255,0,0), 2);
	    
		cv::imshow("detections", Im);
		cv::waitKey();
	}
	cv::destroyWindow("detections");
	*/


	/*
	///////////////////////////////////////
	// test on val subset
	///////////////////////////////////////
	subset = "val";

	cout<<"test on val images"<<endl;
	avgTime = 0;
	for(int i=0; i<valFile.size(); i++){
        cout<<valFile[i];
		string file_addr = dataset + targetFolder + valFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		//detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
		//			modelName, temps, slct_featIdx, gridSize, minThr);
		detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);
		double time_1 = clock();
		double delta_time = (time_1-time_0)/CLOCKS_PER_SEC;
		avgTime += delta_time;
		cout<<" takes "<<delta_time<<" s"<<endl;
		
		string sFileName = valFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_addr = "results/" + subset + "/" + sFileName + "txt";
		FILE *temp_fp = fopen(result_addr.c_str(), "wb");
		for(int j=0; j<found.size(); j++)
			fprintf(temp_fp, "%d %d %d %d %f\n", int(found[j].x), int(found[j].y), int(found[j].width), int(found[j].height), rspn[j]);
		fclose(temp_fp);
		img.release();
		found.clear();
		rspn.clear();
	}
	avgTime /= float(valFile.size());
	cout<<"average time consumption: "<<avgTime<<" s per image"<<endl;
	cout<<endl;
	*/


	
	/*cout<<"evaluation"<<endl;

	fp_det = fopen(("det/" + subset + "_det.txt").c_str(), "wb");
	for(float thr=1; thr>=-1; thr-=0.02){
		float totalObjNum = 0;
		float detectObjNum = 0;
		float falsePosNum = 0;

		string resultAddress = "results/" + subset + "/";
		evaluate_whole(resultAddress, valFile, valGT, true, 0.5, thr, totalObjNum, detectObjNum, falsePosNum);

		float missRate = 1 - detectObjNum/totalObjNum;
		float fppi = falsePosNum/float(valFile.size());
		cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
		fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	}
	fclose(fp_det);
*/
	
	


	/*
	cout<<"display result"<<endl;
    double rspnThr = -0.15; 
	cv::namedWindow("detections");
	for(int i=0; i<valFile.size(); i++){
		string image_address = dataset + targetFolder + valFile[i];
		cv::Mat Im = cv::imread(image_address);
		string sFileName = valFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_address = "res/" + subset + "/" + sFileName + "txt";

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

		for(int j=0; j<valGT[i].size(); j++)
			cv::rectangle(Im, cv::Point(valGT[i][j].x, valGT[i][j].y),
			    cv::Point(valGT[i][j].x+valGT[i][j].width,
				valGT[i][j].y+valGT[i][j].height), CV_RGB(255,0,0), 2);
	    
		cv::imshow("detections", Im);
		cv::waitKey();
	}
	cv::destroyWindow("detections");
	*/


	/*
	///////////////////////////////////////
	// test on unlabel subset
	///////////////////////////////////////
	subset = "unlabel";

	cout<<"test on unlabel images"<<endl;
	
	avgTime = 0;
	for(int i=0; i<unlabelFile.size(); i++){
        cout<<unlabelFile[i];
		string file_addr = dataset + targetFolder + unlabelFile[i];
		cv::Mat img = cv::imread(file_addr);

		vector<cv::Rect> found;
		vector<float> rspn;
		cv::Mat HardNegData;
		double time_0 = clock();
		//detect_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
		//			modelName, temps, slct_featIdx, gridSize, minThr);
		detect_comb_adaboost(img, scaleFactor, staLevId, windSize, winStride, found, rspn, 
					comb_modelName, comb_biasName, temps, slct_featIdx, gridSize, minThr);
		double time_1 = clock();
		double delta_time = (time_1-time_0)/CLOCKS_PER_SEC;
		avgTime += delta_time;
		cout<<" takes "<<delta_time<<" s"<<endl;
		
		string sFileName = unlabelFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_addr = "results/" + subset + "/" + sFileName + "txt";
		FILE *temp_fp = fopen(result_addr.c_str(), "wb");
		for(int j=0; j<found.size(); j++)
			fprintf(temp_fp, "%d %d %d %d %f\n", int(found[j].x), int(found[j].y), int(found[j].width), int(found[j].height), rspn[j]);
		fclose(temp_fp);
		img.release();
		found.clear();
		rspn.clear();
	}
	avgTime /= float(unlabelFile.size());
	cout<<"average time consumption: "<<avgTime<<" s per image"<<endl;
	cout<<endl;
	*/


	/*
	cout<<"evaluation"<<endl;
	
	fp_det = fopen(("det/" + subset + "_det.txt").c_str(), "wb");
	for(float thr=1; thr>=-1; thr-=0.02){
		float totalObjNum = 0;
		float detectObjNum = 0;
		float falsePosNum = 0;

		string resultAddress = "results/" + subset + "/";
		evaluate_whole(resultAddress, unlabelFile, unlabelGT, true, 0.5, thr, totalObjNum, detectObjNum, falsePosNum);

		float missRate = 1 - detectObjNum/totalObjNum;
		float fppi = falsePosNum/float(unlabelFile.size());
		cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
		fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	}
	fclose(fp_det);
	*/

	/*
	cout<<"display result"<<endl;
    double rspnThr = -0.15; 
	cv::namedWindow("detections");
	for(int i=0; i<unlabelFile.size(); i++){
		string image_address = dataset + targetFolder + unlabelFile[i];
		cv::Mat Im = cv::imread(image_address);
		string sFileName = unlabelFile[i];
		sFileName.erase(sFileName.end()-3, sFileName.end());
		string result_address = "res/" + subset + "/" + sFileName + "txt";

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

		for(int j=0; j<unlabelGT[i].size(); j++)
			cv::rectangle(Im, cv::Point(unlabelGT[i][j].x, unlabelGT[i][j].y),
			    cv::Point(unlabelGT[i][j].x+unlabelGT[i][j].width,
				unlabelGT[i][j].y+unlabelGT[i][j].height), CV_RGB(255,0,0), 2);
	    
		cv::imshow("detections", Im);
		cv::waitKey();
	}
	cv::destroyWindow("detections");
	*/


	/*
	string outFolder = "new_proposals_cnn_mit/";
	parseAnnotation(dataset,targetFolder, outFolder, targetNum);

	
	int fileNum = targetFile.size();
	string dataset_name;
	
	
	for(int i=0; i<fileNum; i++){

		if(inWhichSubset(i,fileNum,dataset_name)){
			string fileName = dataset + targetFolder + targetFile[i];
			cv::Mat img = cv::imread(fileName);
			int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols);
			cv::Mat img_border;
			cv::copyMakeBorder(img, img_border, top, top, left, left, cv::BORDER_REPLICATE);
			string out_fileName = outFolder + "images/" + targetFile[i];
			cv::imwrite(out_fileName, img_border);
		}
	}
	

	parseCandidates(dataset, targetFolder, outFolder, targetNum, targetFile, "");
	*/

	
	subset = "test";
	cout<<"evaluation"<<endl;
	
	fp_det = fopen(("det/" + subset + "_det.txt").c_str(), "wb");
	for(float thr=1; thr>=-1; thr-=0.01){
		float totalObjNum = 0;
		float detectObjNum = 0;
		float falsePosNum = 0;

		string resultAddress = "results_cuhk_new/";
		evaluate_full(resultAddress, testFile, testGT, true, 0.5, thr, totalObjNum, detectObjNum, falsePosNum);

		float missRate = 1 - detectObjNum/totalObjNum;
		float fppi = falsePosNum/float(testFile.size());
		cout<<"threshold "<<thr<<"  missRate "<<missRate<<"  fppi "<<fppi<<endl;
		fprintf(fp_det, "%f %f %f\n", thr, missRate, fppi);
	}
	fclose(fp_det);
	

}