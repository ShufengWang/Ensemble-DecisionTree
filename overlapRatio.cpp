#include"common.h"


float overlapRatio(cv::Rect &R1, cv::Rect &R2, int referTo)
{
	float result = 0;
	if((max(R1.x,R2.x)<min(R1.x+R1.width,R2.x+R2.width))&&(max(R1.y,R2.y)<min(R1.y+R1.height,R2.y+R2.height))){
		float overlap_width = min(float(R1.x+R1.width),float(R2.x+R2.width)) - max(float(R1.x),float(R2.x));
		float overlap_height = min(float(R1.y+R1.height),float(R2.y+R2.height)) - max(float(R1.y),float(R2.y));
		float overlap_area = overlap_width*overlap_height;

		float area_1 = float(R1.width)*float(R1.height);
		float area_2 = float(R2.width)*float(R2.height);

		float ratio_0 = overlap_area/(area_1 + area_2 - overlap_area);
		float ratio_1 = overlap_area/area_1;
		float ratio_2 = overlap_area/area_2;
		if(referTo==0){
		    result = ratio_0;
		}
		if(referTo==1){
			result = ratio_1;
		}
		if(referTo==2){
			result = ratio_2;
		}
		if(referTo==3){
			result = max(ratio_1, ratio_2);
		}
	}
	return result;
}