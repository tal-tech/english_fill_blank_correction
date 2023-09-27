#ifndef DETPOSTPROCESS_H
#define DETPOSTPROCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "my_util.h"


class detPostProcess
{
public:
	detPostProcess();
	virtual ~detPostProcess();

	/*
	加载检测结果
	参数：
	IN strDetResult --- 检测模型调用返回得检测结果json数据格式
	OUT vecAllDetResult --- 接收检测结果结构
	返回值：
	201  --- 检测测结果为空(印刷和手写都没有)
	203 --- 检测结果格式错误
	20000  --- 处理正常无错误
	*/
	long loadDetResult(const std::string& strDetResult, std::vector<detResult>& vecAllDetResult);

	/*检测后处理接口,执行部分检测框的合并
	img --- 检测图像brg三通道
	vecAllDetResult  --- 检测结果结构
	vecDetPostProcessResult --- 接收检测后处理结果
	返回值：

	201  --- 检测测结果为空(印刷和手写都没有)
	202 --- 检测后处理错误
	20000  --- 处理正常无错误
	*/
	long getDetPostProcessResult(const cv::Mat& img, const std::vector<detResult>& vecAllDetResult, std::vector<detPostProcessResult>&  vecDetPostProcessResult);

	/*
	取出识别模型需要识别的图像
	参数：
	IN vecDetPostProcessResult  --- 检测后处理结果
	OUT vecEngHWRegIms --- 接收英文手写识别图片
	OUT vecProblemNoRegIms --- 接收题号识别图片
	*/
	void loadRegImgs(const std::vector<detPostProcessResult>&  vecDetPostProcessResult, std::vector<cv::Mat>& vecEngHWRegIms,std::vector<cv::Mat>& vecProblemNoRegIms);
private:
	/*
	图像规范化
	参数：
	IN imgIn --- 待规范化图像
	OUT imgOut --- 规范化后的图像
	IN nNormalWidth --- 规范化图像宽度
	IN nNormalHeight --- 规范化图像高度
	IN nEdgePaddingLen --- 左右边缘padding的像素大小
	*/
	void normalizeImg(const cv::Mat& imgIn, cv::Mat& imgOut, int nNormalWidth, int nNormalHeight,int nType=DET_NO, int nEdgePaddingLen = 16);

	/*
	由点坐标生成mask图像
	参数:
	IN vecMaskDot --- mask图像的点坐标
	IN rtPos --- mask区域的位置
	imgMaskOut --- 接收生成的mask图像
	*/
	void genMaskImg(const std::vector<int>& vecMaskDot, const cv::Rect& rtPos, cv::Mat& imgMaskOut);
};

#endif
