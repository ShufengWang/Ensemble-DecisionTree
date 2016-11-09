#include"common.h"



void load_annotation(string datasetAddress, string testFolderName, int fileNum, vector<string> &fileName, vector<vector<cv::Rect>> &groundTruth)
{
	string annotationFile = datasetAddress + testFolderName + "/annotation.txt";
	FILE *fp = fopen(annotationFile.c_str(), "r");
	for(int i=0; i<fileNum; i++){
		static char tmp[1000];
		fscanf(fp, "%1000s", tmp);
		fileName.push_back((string)tmp);

		int label=0;
		vector<cv::Rect> tempGroundTruth;
		while(1){
			int c;
			do{
				c = getc(fp);
				if(c=='\n')
					goto out2;
			}while(isspace(c));
			ungetc(c, fp);
			int x1, y1, x2, y2;
			fscanf(fp, "%d:%d:%d:%d", &x1, &y1, &x2, &y2);

			cv::Rect tempRect;
			tempRect.x = x1;
			tempRect.y = y1;
			tempRect.width = abs(x2 - x1);
			tempRect.height = abs(y2 - y1);
		    tempGroundTruth.push_back(tempRect);
		}
        out2:
		label=1;

		groundTruth.push_back(tempGroundTruth);
		tempGroundTruth.clear();
	}
	fclose(fp);




}