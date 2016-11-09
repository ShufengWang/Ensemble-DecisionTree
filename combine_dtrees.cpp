#include"common.h"

void combine_dtrees(string& boost_model_file, string prefix, cv::Mat weights)
{

	ofstream ou(boost_model_file);

	char file_name[100];
	prefix += "%d.txt";

	int dtree_num = weights.cols;

	if(!ou.is_open())
		cerr<<"can not open output file "+boost_model_file <<endl;

	for(int i=1;i<=dtree_num;i++){
		if(i>1) 
			ou << "      -"<<endl;
		sprintf(file_name, prefix.c_str(), i);
		string s = file_name;
		ifstream in(s);
		if(!in.is_open())
			cerr<<"can not open input file "+s <<endl;

		string temp;
		getline(in,temp,'\n');
		int flag = 0;
		while(true){

			if(temp.find("ntrees")!=-1){
				if(i==1){
					int len = temp.length();
					for(int j=0;j<len;j++){
						if(temp[j]>='0'&&temp[j]<='9'){
							ou << dtree_num <<endl;
							break;
						}

						ou << temp[j];
					}
				}

				if(!getline(in,temp,'\n')) break;
			}
			else if(temp.find("value")!=-1){
				int len = temp.length();
				bool ok = false;
				for(int j=0;j<len;j++){
					if(!ok){
						ou << temp[j];
						if(temp[j]==':'){
							ok = true;
						}
					}
					else{
						ou << temp[j];
						string xx = string(&temp[j+1]);
						double val;
						sscanf(xx.c_str(),"%lf",&val);
						val *= double(weights.at<float>(0,i-1));
						ou << val <<endl;
						break;
					}
				}
				if(!getline(in,temp,'\n')) break;
			}
			else{
				if(temp.find("nodes")!=-1)
					flag++;
				if(flag||i==1){
					ou << temp <<endl;


				}
				if(!getline(in,temp,'\n')) break;

			}

		}
		in.close();
	}
	ou.close();


}
