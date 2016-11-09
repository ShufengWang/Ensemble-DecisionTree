#include"common.h"




int main()
{

	
	char address[1024];
	
	ifstream fileOut;
	ofstream fileIn;

	srand(unsigned(time(NULL)));


	cv::Size windSize(60,60);

	cv::Size gridSize(6,6);


    vector<cv::Mat> shaarTemps;
	generate_haarTemp_1x1(windSize, gridSize, shaarTemps);
	generate_haarTemp_1x2(windSize, gridSize, shaarTemps);
	generate_haarTemp_1x3(windSize, gridSize, shaarTemps);
	//generate_haarTemp_1x4(windSize, gridSize, shaarTemps);
	generate_haarTemp_2x1(windSize, gridSize, shaarTemps);
	generate_haarTemp_2x2(windSize, gridSize, shaarTemps);
	//generate_haarTemp_2x3(windSize, gridSize, shaarTemps);
	//generate_haarTemp_2x4(windSize, gridSize, shaarTemps);
	generate_haarTemp_3x1(windSize, gridSize, shaarTemps);
	//generate_haarTemp_3x2(windSize, gridSize, shaarTemps);
	generate_haarTemp_3x3(windSize, gridSize, shaarTemps);
	//generate_haarTemp_3x4(windSize, gridSize, shaarTemps);




	// remove duplicates
	vector<cv::Mat> haarTemps;
	haarTemps.push_back(shaarTemps[0]);
	for(int i=1; i<shaarTemps.size(); i++){
		int sign = 0;
		for(int j=0; j<haarTemps.size(); j++)
			if(shaarTemps[i].rows==haarTemps[j].rows){
				cv::Mat temp = abs(shaarTemps[i]-haarTemps[j]);
				if(sum(temp)[0]<1){
					sign = 1;
					break;
				}
			}
		if(sign==0)
			haarTemps.push_back(shaarTemps[i]);
	}
	cout<<"# temps: "<<haarTemps.size()<<endl;



	// save templates
	for(int i=0; i<haarTemps.size(); i++){
		sprintf(address, "%s%d%s", "haarTemps/t_", i, ".txt");
		fileIn.open(address);
		for(int j=0; j<haarTemps[i].rows; j++){
			for(int k=0; k<haarTemps[i].cols; k++)
				fileIn<<haarTemps[i].at<float>(j,k)<<" ";
			fileIn<<"\n";
		}
		fileIn.close();
	}




	// display
	string figName = "template";
	cv::namedWindow(figName);
	for(int i=0; i<haarTemps.size(); i++){
		cv::Mat tempWin(windSize.height, windSize.width, CV_32F, cv::Scalar(0));
		for(int j=0; j<haarTemps[i].rows; j++){
			if(haarTemps[i].at<float>(j,2)>0)
				tempWin(cv::Rect(haarTemps[i].at<float>(j,0)*gridSize.width, haarTemps[i].at<float>(j,1)*gridSize.height, gridSize.width, gridSize.height))=1.0;
			if(haarTemps[i].at<float>(j,2)<0)
				tempWin(cv::Rect(haarTemps[i].at<float>(j,0)*gridSize.width, haarTemps[i].at<float>(j,1)*gridSize.height, gridSize.width, gridSize.height))=0.3;
	    }
	    tempWin.convertTo(tempWin, CV_8U, 255.0, 0.0);
	    cv::imshow(figName, tempWin);
	    cv::waitKey(50);
	}
	cv::destroyWindow(figName);



	return 0;
}