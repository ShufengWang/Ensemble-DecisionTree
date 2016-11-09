#include"common.h"

void imageToVideo(string dataset, string targetFolder, string subFolder, vector<string> targetFile)
{
	cv::VideoWriter fileWriter;
	fileWriter.open(dataset + targetFolder + subFolder + "detection_results.avi", CV_FOURCC('D','I','V','X'), 2.0, cv::Size(720,528), true);
	for(int i=0; i<targetFile.size(); i++){
		string sFileName = targetFile[i];
		string t_add = dataset + targetFolder + subFolder + sFileName;
		cv::Mat Im = cv::imread(t_add);
		cv::resize(Im, Im, cv::Size(720,528));
		fileWriter.write(Im);
	}
	fileWriter.release();

}