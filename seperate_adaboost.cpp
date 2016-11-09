#include"common.h"

void seperate_adaboost(string& boost_model_file, string save_prefix, int dtree_id)
{

	ifstream in(boost_model_file);

	char file_name[100];
	save_prefix += "%d.txt";
	sprintf(file_name, save_prefix.c_str(), dtree_id);
	string s = file_name;
	ofstream ou(s);

	if(!in.is_open())
		cerr<<"can not open input file"<<endl;
	if(!ou.is_open())
		cerr<<"can not open output file"<<endl;

	//char temp[10000];
	string temp;
	getline(in,temp,'\n');
	int flag = 0;
	while(true){

		if(temp.find("ntrees")!=-1){
			int len = temp.length();
			for(int i=0;i<len;i++){
				if(temp[i]>='0'&&temp[i]<='9'){
					ou <<"1\n";
					break;
				}
				//printf("%c",temp[i]);
				ou << temp[i];
			}
			if(!getline(in,temp,'\n')) break;
		}
		else{
			if(temp.find("nodes")!=-1)
				flag++;
			if(flag==0){
				ou << temp <<endl;
				if(!getline(in,temp,'\n')) break;
				continue;
			}
			if(flag>dtree_id) break;
			string nxt;

			if(!getline(in,nxt,'\n')){
				ou << temp <<endl;
				break;
			}
			if(flag==dtree_id&&nxt.find("nodes")==-1)
				ou << temp <<endl;

			temp = nxt;

		}

	}
	in.close();
	ou.close();

}
