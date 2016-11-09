#include"common.h"



void load_templates(char *address, cv::Mat &haarTemp)
{

	FILE *fp = fopen(address, "r");
	if(fp==NULL){
		fprintf(stderr, "can't open input file \"%s\"\n", address);
		exit(1);
	}

	for(int i=0; i<100; i++){
		int fId;

		while(1){
			int c;
			do{
				c = getc(fp);
				if((c=='\n')||(c==-1))
					goto out2;
			}while(isspace(c));
			ungetc(c,fp);

			float x, y;
			int labl;
			fscanf(fp, "%g %g %d", &x, &y, &labl);

			cv::Mat temp(1, 3, CV_32F, cv::Scalar(0));
			temp.at<float>(0,0) = x;
			temp.at<float>(0,1) = y;
			temp.at<float>(0,2) = labl;
			haarTemp.push_back(temp);
			temp.release();
		}
		out2:fId=1;
	}

	fclose(fp);

}

