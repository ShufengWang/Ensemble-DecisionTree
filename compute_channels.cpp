#include"common.h"


void compute_channels(cv::Mat img, vector<cv::Mat> &channels)
{
	
    Mat_<float> grad;
    Mat_<float> angles;
    Mat luv, gray, src;
    
    src = Mat(img.rows, img.cols, CV_32FC3);
    img.convertTo(src, CV_32FC3, 1./255);

    cvtColor(src, gray, CV_RGB2GRAY);
    cvtColor(src, luv, CV_RGB2Luv);

    Mat_<float> row_der, col_der;
    Sobel(gray, row_der, CV_32F, 0, 1);
    Sobel(gray, col_der, CV_32F, 1, 0);

    cartToPolar(col_der, row_der, grad, angles, true);
    //magnitude(row_der, col_der, grad);

    Mat_<Vec6f> hist = Mat_<Vec6f>::zeros(grad.rows, grad.cols);
    //const float to_deg = 180 / 3.1415926f;
    for (int row = 0; row < grad.rows; ++row) {
        for (int col = 0; col < grad.cols; ++col) {
            //float angle = atan2(row_der(row, col), col_der(row, col)) * to_deg;
            float angle = angles(row, col);
            if (angle < 0)
                angle += 180;
            int ind = (int)(angle / 30);

            // If angle == 180, prevent index overflow
            if (ind == 6)
                ind = 5;

            hist(row, col)[ind] = grad(row, col) * 255;
        }
    }

    channels.clear();

	Mat luv_channels[3];
	split(luv, luv_channels);
	for( int i = 0; i < 3; ++i )
		channels.push_back(luv_channels[i]);

    channels.push_back(grad);

    vector<Mat> hist_channels;
    split(hist, hist_channels);

    for( size_t i = 0; i < hist_channels.size(); ++i )
        channels.push_back(hist_channels[i]);


}