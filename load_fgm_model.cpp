#include"common.h"



void load_fgm_model(char *address, vector<int> &featIdx, vector<float> &coe)
{
    FILE *fp = fopen(address,"r");
    if(fp == NULL)
    {
		fprintf(stderr,"Can't open input file \"%s\"\n", address);
		exit(1);
    }

    static char tmp[1000];
    fscanf(fp, "%1000s", tmp); //solver_type
    fscanf(fp, "%1000s", tmp); //L1R_L2LOSS_SVC

    fscanf(fp, "%1000s", tmp); //nr_class
    fscanf(fp, "%1000s", tmp); //2

	fscanf(fp, "%1000s", tmp); // label
    fscanf(fp, "%1000s", tmp); // 1
    fscanf(fp, "%1000s", tmp); // -1

    fscanf(fp, "%1000s", tmp); // nr_feature
    fscanf(fp, "%1000s", tmp); // 

    fscanf(fp, "%1000s", tmp); // bias
	fscanf(fp, "%1000s", tmp); // -1

	fscanf(fp, "%1000s", tmp); // B
	fscanf(fp, "%1000s", tmp); // 2

	fscanf(fp, "%1000s", tmp); // flag_poly
	fscanf(fp, "%1000s", tmp); // 0

	fscanf(fp, "%1000s", tmp); // coef0
	fscanf(fp, "%1000s", tmp); // 1

	fscanf(fp, "%1000s", tmp); // gamma
	fscanf(fp, "%1000s", tmp); // 1

	fscanf(fp, "%1000s", tmp); // t
	fscanf(fp, "%1000s", tmp); // 1

	fscanf(fp, "%1000s", tmp); // feature_pair
	int featureNum;
	fscanf(fp, "%d", &featureNum); // 

	fscanf(fp, "%1000s", tmp); // train_time
	fscanf(fp, "%1000s", tmp); // 

	fscanf(fp, "%1000s", tmp); // w
	fscanf(fp, "%1000s", tmp); // 

	
    // now load feature index ...
    for(int i=0; i<featureNum; i++){
		int fId;
		fscanf(fp, "%d", &fId);
		featIdx.push_back(fId);

		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') 
					goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			int index;
			double value;
			fscanf(fp, "%d:%lf", &index, &value);
			coe.push_back(value);
		}	
        out2:
		fId=1; // dummy
    }

	
    // finished!
    fclose(fp);

}