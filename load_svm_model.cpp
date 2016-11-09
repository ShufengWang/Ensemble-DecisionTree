#include"common.h"



void load_svm_model(char *address, int feaDim, cv::Mat &alpha, float &rho, cv::Mat &supVec)
{
    FILE *fp = fopen(address,"r");
    if(fp == NULL)
    {
		fprintf(stderr,"Can't open input file \"%s\"\n", address);
		exit(1);
    }

    static char tmp[1000];
    fscanf(fp, "%1000s", tmp); //svm_type
    fscanf(fp, "%1000s", tmp); //c_svc
    fscanf(fp, "%1000s", tmp); //kernel_type
    fscanf(fp, "%1000s", tmp); //linear

    fscanf(fp, "%1000s", tmp); // nr_class
    fscanf(fp, "%1000s", tmp); // 2
    fscanf(fp, "%1000s", tmp); // total_sv

	int supVecNum;
    fscanf(fp, "%d", &supVecNum); 
	cv::Mat temp_alpha(1, supVecNum, CV_32FC1, cv::Scalar(0));
	cv::Mat temp_supVec(supVecNum, feaDim, CV_32FC1, cv::Scalar(0));

    fscanf(fp, "%1000s", tmp); //rho
    fscanf(fp, "%f\n", &rho);

    fscanf(fp, "%1000s", tmp); // label
    fscanf(fp, "%1000s", tmp); // 1
    fscanf(fp, "%1000s", tmp); // -1
    fscanf(fp, "%1000s", tmp); // nr_sv
    fscanf(fp, "%1000s", tmp); // num
    fscanf(fp, "%1000s", tmp); // num
    fscanf(fp, "%1000s", tmp); // SV
	

    // now load SV data...
	int index, oldindex=0;
    for(int i=0; i<supVecNum; i++){
		double label;
		fscanf(fp, "%lf", &label);
		temp_alpha.at<float>(0,i) = label;

		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') 
					goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			double value;
			fscanf(fp, "%d:%lf", &index, &value);
			if(index!=oldindex)
			{
				temp_supVec.at<float>(i, index-1) = value;
			}
			oldindex=index;
		}	
		out2:
		label=1; // dummy
    }

	
    // finished!
    fclose(fp);

	temp_alpha.copyTo(alpha);
	temp_supVec.copyTo(supVec);

}