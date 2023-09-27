#ifndef UTIL_H
#define UTIL_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define REG_HW_IMG_H 64  //英文手写图片规范化高度
#define REG_HW_IMG_W 1024   //英文手写图片规范化宽度
#define REG_NO_IMG_H 64    //题号图片规范化高度
#define REG_NO_IMG_W 768   //题号图片规范化宽度

#define MAX_RESULT_OPT_SIZE  10  //候选结果最大个数

enum err_type
{
	ERR_DETPOSTPROCESS_NO_DET_RESULT = 201, //检测后处理没有检测结果
	ERR_DETPOSTPROCESS_ERROR = 202, //检测后处理错误
	ERR_DETPOSTPROCESS_WORONG_DET_FORMAT = 203, //检测结果格式错误
	ERR_REGPOSTPROCESS_NO_REG_RESULT = 501,//识别后处理没有识别结果输入
	ERR_REGPOSTPROCESS_ERROR = 502,  //识别后处理错误
	ERR_REGPOSTPROCESS_DET_REG_ERROR = 503, //识别与检测有错误
	ERR_REGPOSTPROCESS_DET_REG_NOMATCH = 504, //识别与检测不匹配
	ERR_REGPOSTPROCESS_WRONG_ANSWER_FORMAT= 505,//识别后处理标准答案为空
	ERR_REGPOSTPROCESS_EMPTY_ANSWER= 506, //识别后处理标准答案为空

	ERR_OK = 20000   //处理成功
};
//检测类别枚举
enum det_type
{
	DET_NO = 0,//题号
	DET_HWContent //手写内容
};
//检测结果
typedef struct tagDetResult
{
	cv::Rect rtDet;  //矩形框位置
	std::vector<int> vecMaskDots;  //检测框mask区域
	int nClass;//类别
	float fProb;//置信度
}detResult;

//检测后处理结果
typedef struct tagDetPostProcessResult
{
	cv::Mat matImg2Reg; //待识别图像
	std::vector<cv::Rect> vecCombiOriPos; //小图在大图的位置，存在多张小图合成一张识别图的情况
	std::vector<int> vecCombiStartPos; //合成图中，每张小图的的起始位置
	int nType; //类型 0---题号，  1--- 英语手写
}detPostProcessResult;

typedef struct tagRegResult
{
	std::vector<std::string> vecCondidateResult; //识别候选结果
	std::vector<std::vector<int> > vecCondidateResultPos; //所有候选结果的每个字符的位置，需要对应到识别输入的原图
	std::vector<float> vecProb;//每个候选结果的识别置信度
	int nType; //类型 0---题号，  1--- 英语手写
}regResult;
//识别后处理结果
typedef struct tagRegPostProcessResult
{
	std::string strNo; //题号
	std::vector<std::string> vecRegResult; //手写多个候选结果
	std::vector<float> vecRegProb; //识别置信度
	cv::Rect rtPos; //题目在原图位置
	std::string strAnswer; //标准答案
	int nJudgeResult; //判题结果
	float fJudgeProb; //判题置信度
}regPostProcessResult;

//标准答案结构
typedef struct tagAnswer
{
	std::string strNo; //题号
	std::string strAnswer; //答案
}answer;


#endif
