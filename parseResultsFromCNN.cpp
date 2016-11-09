#include"common.h"


void parseResultFromCNN(string dataset, string targetFolder, string subset){

	string inputFile = "res_cnn/det_"+subset+"_Pedestrian.txt";
	FILE *fp = fopen(inputFile.c_str(), "r");
	static char tmp[1000];
	char lastFile[1000];
	ofstream ou;
	string outputFile;
	while(fscanf(fp, "%s", tmp)!=EOF){
		if(strcmp(tmp,lastFile)!=0){
			outputFile = "res_cnn/"+subset+"_2/"+string(tmp)+".txt";
			ou = ofstream(outputFile);
		}
		cv::Mat img = cv::imread(dataset + targetFolder +string(tmp)+".png");
		int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols); 
		int x1,y1,x2,y2;
		float sc;
		fscanf(fp,"%f%d%d%d%d",&sc,&x1,&y1,&x2,&y2);
		ou<<x1-left<<" "<<y1-top<<" "<<x2-x1<<" "<<y2-y1<<" "<<sc<<endl;
		strcpy(lastFile,tmp);
	}
	fclose(fp);
}