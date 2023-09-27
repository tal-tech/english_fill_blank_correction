#ifndef ENGREGINTERFACE_H
#define ENGREGINTERFACE_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "my_util.h"
class Model_English;

class Classifier_English
{
public:
          Classifier_English();
	 virtual ~Classifier_English();
          /*
	   *初始化模型类
	   * 参数:
	   *IN model_file --- 模型定义文件.prototxt文件
	   * IN trained_file --- 权值文件 .caffemodel文件
	   * IN gpu --- 指定的显卡序号，默认-1即不需要指定具体的显卡(比如paas平台)
	   * 返回值：
	   * 401 --- 模型定义.prototxt文件加载失败
	   * 402  --- 模型权值 .caffemodel文件加载失败
	   * 404 --- 模型初始化失败
	   * 20000  --- 成功
	   *                                                                                         */
	  int Init(const std::string& model_file, const std::string& trained_file, int gpu = -1);
          Model_English* getModel();
    private:
	 Model_English* m_model;
};


class EngRegInterface
{
public:
	EngRegInterface() ;
	~EngRegInterface();
    /*
	英文手写识别接口
	IN RegImg --- 待识别的小图
	OUT RegResult --- 接收识别结果
	IN regModel ---- 识别模型指针
	IN str_true_label  --- 识别的真实内容(标准结果), 默认传空字符串
	返回值：
	3006210405 --- 输入的图像为空
	3006210407 --- 输入的模型为空指针
	3006210408 --- 识别出错
	20001 --- 识别成功
	*/
	int getRegResult(const cv::Mat& RegImg, regResult& RegResult, Classifier_English *regModel,const std::string& str_true_label="");

};
#endif 

