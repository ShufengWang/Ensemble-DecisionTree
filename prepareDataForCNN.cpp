#include"common.h"


void parseAnnotation(string datasetAddress, string testFolderName, string outFolder, int fileNum){
	string annotationFile = datasetAddress + testFolderName + "annotation.txt";
	FILE *fp = fopen(annotationFile.c_str(), "r");
	string outDir = outFolder + "annotations_txt/";
	for(int i=0; i<fileNum; i++){
		static char tmp[1000];
		fscanf(fp, "%1000s", tmp);

		char image_id[1000];
		int j;
		for(j=0;tmp[j];j++){
			if(tmp[j]=='.') break;
			image_id[j] = tmp[j];
		}
		image_id[j] = '\0';
		ofstream ou;
		string dataset_name;
		bool flag = inWhichSubset(i, fileNum, dataset_name);
		
		if(flag)
			ou = ofstream((outDir + dataset_name + "/" + image_id + ".txt").c_str());

		
		cv::Mat img = cv::imread(datasetAddress+testFolderName+tmp);
		int top = int(0.2*img.rows), left = int(0.2*img.cols);
		int label=0;
		
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
			int w = x2-x1;
			int h = y2-y1;
			x1 = x1+0.5*w-0.3125*h+left;
			y1 = y1-0.125*h+top;
			x2 = x1+0.625*h;
			y2 = y1+1.25*h;
			if(flag)
				ou<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
		    
		}
        out2:
		label=1;

		
		
	}
	fclose(fp);

	


}
bool inWhichSubset(int i,int fileNum, string & dataset_name){
	bool flag = false;
	if(i>=420){
		//if(i%4==0)
		dataset_name = "test", flag = true;
	}
	else if(i%20==0){
		dataset_name = "train", flag = true;
	}
	else if(i%20==10)
		dataset_name = "val", flag = true;	
	else
		dataset_name = "unlabel", flag = true;
	
	return flag;
}
void parseCandidates(string datasetAddress, string testFolderName, string outFolder, int fileNum, vector<string> tempFile, string targetSubset)
{
	//parseCandidates
	
	string result_dir = "results/";
	
	string dataset_name = "";
	
	for(int i=0; i<fileNum; i++){
		if(inWhichSubset(i,fileNum,dataset_name)&&(targetSubset==""||targetSubset.compare(dataset_name)==0)){
				string fileName = datasetAddress + testFolderName + tempFile[i];
				cv::Mat img = cv::imread(fileName);
				int top = (int)(0.2*img.rows), left = (int)(0.2*img.cols);
				char image_id[1000];
				int j;
				for(j=0;tempFile[i][j];j++){
					if(tempFile[i][j]=='.') break;
					image_id[j] = tempFile[i][j];
				}
				image_id[j] = '\0';
				
				
				//string res_fileName = result_dir + dataset_name + "/"+ image_id + ".txt";
				string res_fileName = result_dir + image_id + ".txt";


				FILE *fp = fopen(res_fileName.c_str(), "r");
				string outDir = outFolder + "candidates_txt/";
				
				int x1,y1,h,w;
				float score;
				ofstream ou((outDir + dataset_name + "/" + image_id + ".txt").c_str());

				while(fscanf(fp,"%d%d%d%d%f",&x1,&y1,&w,&h,&score)!=EOF){
					x1 += left;
					y1 += top;
					//ou << x1 <<" "<< y1 <<" "<< x1+w <<" "<< y1+h <<endl;
					ou << x1 <<" "<< y1 <<" "<< x1+w <<" "<< y1+h <<" "<< score <<endl;
				}
				fclose(fp);
				
				

		    }
			
	}
}