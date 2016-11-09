#include"common.h"

void AdaBoost(cv::Mat trainData, cv::Mat response, int maxDepth, int weakCount, string modelName)
{


	Mat var_type( 1, trainData.cols+1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(trainData.cols) = VAR_CATEGORICAL;

	Ptr<TrainData> tdata = TrainData::create(trainData, ROW_SAMPLE, response, noArray(), noArray(), noArray(), var_type);

	Ptr<Boost> model;
	model = Boost::create();
	model->setBoostType(Boost::GENTLE);
	model->setCVFolds(0);
	model->setMaxDepth(maxDepth);
	model->setWeakCount(weakCount);
	model->setWeightTrimRate(0.95);
	model->train(tdata);
	model->save(modelName);

	model->clear();
	tdata->~TrainData();


}