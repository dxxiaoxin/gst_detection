#include<iostream>
#include<opencv2\opencv.hpp>
#include<math.h>
#include <vector>
#include<algorithm>
#include"ffttools.hpp"

using namespace cv;
using namespace std;
#define pi 3.14159265358



enum ConvolutionType                     // 函数 conv2 卷积时参数的类型
{
    CONVOLUTION_FULL,                    // 卷积时的参数，和 matlab 的 full 一致
	CONVOLUTION_SAME,                    // 卷积时的参数，和 matlab 的 same 一致
	CONVOLUTION_VALID                    // 卷积时的参数，和 matlab 的 valid 一致
 };


/***************************************
* xgv -- 【输入】指定X输入范围
* ygv -- 【输入】指定Y输入范围
* X   -- 【输出】Mat
* Y   -- 【输出】Mat
****************************************/
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<float> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
/*
function h = Ffilter(n,sigma, filterSize)
% compute the symmetry derivative filter of 2D Gaussian"
mdpt = ceil(filterSize/2);
[x y] = meshgrid(-mdpt:mdpt, -mdpt:mdpt);
if n > 0
h = (-1/(sigma*sigma))^n /(2*pi*sigma*sigma) * ...
(x+i*y).^n .* exp(-(x.*x+y.*y)/(2*sigma*sigma));
else
n = -n;
h = (-1/(sigma*sigma))^n /(2*pi*sigma*sigma) * ...
(x-i*y).^n .* exp(-(x.*x+y.*y)/(2*sigma*sigma));
end
h = h/sum(abs(h(:)));
*/
Mat Ffilter(int n, float sigma, int filterSize)
{
	Mat h;
	Mat F[2];
	Mat res[2];
	double hPart1;
	double hPart2;
	Mat hPart3;
	int mdpt = ceil(filterSize / 2.0);
	Mat x, y;
	meshgrid(cv::Range(-mdpt, mdpt), cv::Range(-mdpt, mdpt), x, y);

	cv::exp(-(x.mul(x) + y.mul(y)) / (2 * sigma*sigma),hPart3);



	if (n > 0)
	{
		hPart1 = pow((-1 / (sigma*sigma)), n);
		hPart2 = hPart1 / (2 * pi*sigma*sigma);
		cv::pow(x, n, F[0]);
		cv::pow(y, n, F[1]);

		
		res[0] = (hPart2*F[0]).mul(hPart3);
		res[1] = (hPart2*F[1]).mul(hPart3);



		merge(res, 2, h);

	}
	else
	{
		n = -n;
		hPart1 = pow((-1 / (sigma*sigma)), n);
		hPart2 = hPart1 / (2 * pi*sigma*sigma);

		Mat ac;
		multiply(x, x, ac);//multiply实现矩阵的点乘
		Mat bd;
		multiply(y, y, bd);
		Mat Real = ac - bd;//得到的结果的实部
		Mat ad;
		multiply(x, y, ad);
		Mat bc;
		multiply(y, x, bc);
		Mat Im = -(ad + bc);
		Mat value[2] = { Real,Im };

		res[0] = (hPart2*value[0]).mul(hPart3);
		res[1] = (hPart2*value[1]).mul(hPart3);

		//printf("\n");
		//for (int i = 0; i < value[0].rows; i++)
		//{
		//	for (int j = 0; j < value[0].cols; j++)
		//	{
		//		printf("%.8f ", value[0].at<float>(i, j));
		//	}
		//	printf("\n");
		//}
		//printf("\n");
		//for (int i = 0; i < value[1].rows; i++)
		//{
		//	for (int j = 0; j < value[1].cols; j++)
		//	{
		//		printf("%.8f ", value[1].at<float>(i, j));
		//	}
		//	printf("\n");
		//}

		merge(res, 2, h);




	}
	
	float sumVal = 0.0;

	for (int i = 0; i < res[0].rows; i++)
	{
		for (int j = 0; j < res[0].cols; j++)
		{
			sumVal += sqrtf(powf(abs(res[0].ptr<float>(i)[j]), 2) + powf(abs(res[1].ptr<float>(i)[j]), 2));
		}
	}

	//printf("\n");
	//for (int i = 0; i < h.rows; i++)
	//{
	//	for (int j = 0; j < h.cols; j++)
	//	{
	//		printf("%.8f ", h.at<Vec2f>(i,j)[0]);
	//	}
	//	printf("\n");
	//}

	printf("\n");
	h = h / sumVal;
	
	//for (int i = 0; i < h.rows; i++)
	//{
	//	for (int j = 0; j < h.cols; j++)
	//	{
	//		printf("%.8f ", h.at<Vec2f>(i, j)[0]);
	//	}
	//	printf("\n");
	//}

	return h;
}

/*
function result = reviseImageBoundary(I, dw)
% set the image boundary to 0
[m n] = size(I);
I = double(I);
result = zeros(m, n);
result(dw:m-dw, dw:n-dw) = I(dw:m-dw, dw:n-dw);


*/
Mat reviseImageBoundary(Mat I, int dw)
{
	Mat res = Mat::zeros(Size(I.cols, I.rows), CV_32FC1);

	for (int i = dw; i < I.rows - dw; i++)
	{
		for (int j = dw; j < I.cols - dw; j++)
		{
			res.ptr<float>(i)[j] = I.ptr<float>(i)[j];
		}
	}
	return res;
}



double comMax(double x, double y)
{
	return x > y ? x : y;
}

/*
function bw = segment_simple_threshold(I)
% use simple threshold to segment the image
T_min = 0.000002;
T = max(T_min, 0.4 * max(I(:)));
bw = uint8( 255 * (I>T) );
*/
Mat segment_simple_threshold(Mat I)
{
	Mat res = Mat::zeros(Size(I.cols,I.rows),CV_8UC1);
	double T_min = 0.000002;
	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(I, NULL, &maxv,NULL,NULL);
	double T = comMax(T_min, 0.01 * maxv);     //默认0.4
	for (int i = 0; i < I.rows; i++)
	{
		for (int j = 0; j < I.cols; j++)
		{
			if (I.ptr<float>(i)[j] > T)
			{
				res.ptr<uchar>(i)[j] = 255;
			}
		}
	}
	return res;

}

/*
二维复数卷积


*/

Mat conv2(const Mat &img, const Mat& ikernel, ConvolutionType type)
{
	//Mat dest;
	Mat res;
	Mat kernel;
	flip(ikernel, kernel, -1);
	Mat source = img;
	if (CONVOLUTION_FULL == type)
	{
		source = Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;

	//printf("kernel:%d\n", kernel.channels());

	if (kernel.channels() == 1)
	{
		Mat dest;
		filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);
		res = dest.clone();
	}
	else
	{
		std::vector<Mat> kernelArr;
		Mat dest[2];
		split(kernel, kernelArr);


		filter2D(source, dest[0], img.depth(), kernelArr[0], anchor, 0, borderMode);
		filter2D(source, dest[1], img.depth(), kernelArr[1], anchor, 0, borderMode);

		merge(dest, 2, res);
	}

	
	return res;
}

#if 0
int main()
{
	Mat F1,h;
	Mat f1,Z,I20,I10,CtPart1,CtPart2,Ct, revised_Ct, bw;
	

	float sigma1 = 0.6;
	float sigma2 = 1.1;
	int filterSize = 5;
	int boundaryWidth = 5; // the boundary width
	h = Ffilter(-2, sigma2, filterSize);
	F1 = Ffilter(1, sigma1, filterSize);

	VideoCapture cap;
	//cap.open("F:\\dataSet\\sd2 截取视频.avi");
	//cap.open("F:\\dataSet\\yx视频分析\\2022_08_19-17_38_12.mp4");
	cap.open("F:\\dataSet\\kcf测试用视频\\20200831172411.mkv");

	if (!cap.isOpened())
	{
		return -1;
	}
	Mat input;

	long frame_num = (long)cap.get(cv::CAP_PROP_FRAME_COUNT);
	long cnt = 0;
	int cntTrack = 0;
	// get bounding box
	cap >> input;

	Rect inputRoi;
	inputRoi.x = 10;
	inputRoi.y = 10;
	inputRoi.width = input.cols - 20;
	inputRoi.height = input.rows - 20;

	Mat img;
	while (cnt < frame_num)
	{
		cap >> input;
		cnt++;
		cv::cvtColor(input, img, cv::COLOR_RGB2GRAY);
		//Mat img = imread("F:\\PCLProjects\\GST-for-small-target-detection-master\\images\\02.bmp",0);
		//Mat img = imread("1.jpg", 0);

		img = img(Rect(10, 10, img.cols - 20, img.rows - 20));

		imshow("img", img);

		img.convertTo(img, CV_32FC1);

		clock_t start = clock();

		//if (img.channels() == 1)
		//{
		//	cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
		//	//cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
		//	cv::merge(planes, 2, img);
		//}



		//printf("img:%d\n", img.channels());
		//printf("F1:%d\n", F1.channels());
		f1 = conv2(img, F1, CONVOLUTION_SAME);





		//printf("f1:%d\n", f1.channels());

		Z = (FFTTools::complexMultiplication(f1, f1)) / 255.0;

		std::vector<cv::Mat> ZArr;
		std::vector<cv::Mat> hArr;
		split(h, hArr);
		split(Z, ZArr);
		Mat ac;
		ac = conv2(ZArr[0], hArr[0], CONVOLUTION_SAME);//multiply实现矩阵的点乘
		Mat bd;
		bd = conv2(ZArr[1], hArr[1], CONVOLUTION_SAME);
		Mat Real = ac - bd;//得到的结果的实部
		Mat ad;
		ad = conv2(ZArr[0], hArr[1], CONVOLUTION_SAME);
		Mat bc;
		bc = conv2(ZArr[1], hArr[0], CONVOLUTION_SAME);
		Mat Im = ad + bc;
		Mat value[2] = { Real,Im };

		merge(value, 2, I20);

		//I20 = twoDConvFunc(Z, h, CONVOLUTION_SAME);





		I10 = conv2(FFTTools::real(FFTTools::magnitude(Z)), FFTTools::real(FFTTools::magnitude(h)), CONVOLUTION_SAME);




		Mat coscs = Mat::zeros(Size(I20.cols, I20.rows), CV_32FC1);

		std::vector<cv::Mat> planesI20;
		split(I20, planesI20);

		for (int i = 0; i < I20.rows; i++)
		{
			for (int j = 0; j < I20.cols; j++)
			{
				coscs.ptr<float>(i)[j] = cos(atan2(planesI20[1].ptr<float>(i)[j], planesI20[0].ptr<float>(i)[j]));
			}
		}


		//printf("CtPart1:%d\n", CtPart1.channels());
		//printf("I10:%d\n", I10.channels());
		CtPart1 = FFTTools::magnitude(I20);
		CtPart2 = CtPart1.mul(I10);
		Ct = CtPart2.mul(1 + coscs);



		revised_Ct = reviseImageBoundary(Ct, boundaryWidth);



		bw = segment_simple_threshold(revised_Ct);

		clock_t end = clock();
		double time = (end - start) / 1000.0;

		printf("time is :%.4f", time);

		imshow("bw", bw);
		waitKey(10);
	}
	return 0;
}
#endif
#if 1
int main()
{
	Mat F1, h;
	Mat f1, Z, I20, I10, CtPart1, CtPart2, Ct, revised_Ct, bw;


	float sigma1 = 0.6;
	float sigma2 = 1.1;
	int filterSize = 3;
	int boundaryWidth = 5; // the boundary width
	h = Ffilter(-2, sigma2, filterSize);
	F1 = Ffilter(1, sigma1, filterSize);
	string pattern_tif = "D:\\车辆数据集\\Images\\50\\*.bmp";
	vector<cv::String> image_files;
	glob(pattern_tif, image_files, false);
	int cnt = 0;
	while (1)
	{
		
		Mat img = imread(image_files[cnt], 0);
		
		//Mat img = imread("F:\\PCLProjects\\GST-for-small-target-detection-master\\images\\02.bmp",0);
		//Mat img = imread("1.jpg", 0);
		if (img.empty())
			break;
		cnt++;
		img = img(Rect(10, 10, img.cols - 20, img.rows - 20));

		imshow("img", img);

		img.convertTo(img, CV_32FC1);

		clock_t start = clock();

		//if (img.channels() == 1)
		//{
		//	cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
		//	//cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
		//	cv::merge(planes, 2, img);
		//}



		//printf("img:%d\n", img.channels());
		//printf("F1:%d\n", F1.channels());
		f1 = conv2(img, F1, CONVOLUTION_SAME);





		//printf("f1:%d\n", f1.channels());

		Z = (FFTTools::complexMultiplication(f1, f1)) / 255.0;

		std::vector<cv::Mat> ZArr;
		std::vector<cv::Mat> hArr;
		split(h, hArr);
		split(Z, ZArr);
		Mat ac;
		ac = conv2(ZArr[0], hArr[0], CONVOLUTION_SAME);//multiply实现矩阵的点乘
		Mat bd;
		bd = conv2(ZArr[1], hArr[1], CONVOLUTION_SAME);
		Mat Real = ac - bd;//得到的结果的实部
		Mat ad;
		ad = conv2(ZArr[0], hArr[1], CONVOLUTION_SAME);
		Mat bc;
		bc = conv2(ZArr[1], hArr[0], CONVOLUTION_SAME);
		Mat Im = ad + bc;
		Mat value[2] = { Real,Im };

		merge(value, 2, I20);

		//I20 = twoDConvFunc(Z, h, CONVOLUTION_SAME);





		I10 = conv2(FFTTools::real(FFTTools::magnitude(Z)), FFTTools::real(FFTTools::magnitude(h)), CONVOLUTION_SAME);




		Mat coscs = Mat::zeros(Size(I20.cols, I20.rows), CV_32FC1);

		std::vector<cv::Mat> planesI20;
		split(I20, planesI20);

		for (int i = 0; i < I20.rows; i++)
		{
			for (int j = 0; j < I20.cols; j++)
			{
				coscs.ptr<float>(i)[j] = cos(atan2(planesI20[1].ptr<float>(i)[j], planesI20[0].ptr<float>(i)[j]));
			}
		}


		//printf("CtPart1:%d\n", CtPart1.channels());
		//printf("I10:%d\n", I10.channels());
		CtPart1 = FFTTools::magnitude(I20);
		CtPart2 = CtPart1.mul(I10);
		Ct = CtPart2.mul(1 + coscs);



		revised_Ct = reviseImageBoundary(Ct, boundaryWidth);



		bw = segment_simple_threshold(revised_Ct);

		clock_t end = clock();
		double time = (end - start) / 1000.0;

		printf("time is :%.4f", time);

		imshow("bw", bw);
		waitKey(0);

	}
	
	return 0;
}

#endif