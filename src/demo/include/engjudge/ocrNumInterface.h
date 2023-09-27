#ifndef OCRNUMINTERFACE_H
#define OCRNUMINTERFACE_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "my_util.h"

class Model_Number;

class ocrNumRegModel
{
public:
    ocrNumRegModel();
    virtual ~ocrNumRegModel();
/*
* *   *初始化模型类
*     * 参数:
*    *IN model_file --- 模型定义文件.prototxt文件
*   * IN trained_file --- 权值文件 .caffemodel文件
*   * IN gpu --- 指定的显卡序号，默认-1即不需要指定具体的显卡(比如paas平台)
*   * 返回值：
*   * 301 --- 模型定义.prototxt文件加载失败
*   * 302  --- 模型权值 .caffemodel文件加载失败
*   * 303 --- gpu显卡设定存在问题
*   * 304 --- 模型初始化失败
*   * 20000  --- 成功
*   *   
*   */
    int Init(const std::string& model_file, const std::string& trained_file, int gpu = -1);
    Model_Number* getModel();
private:
    Model_Number* m_model;
};

class ocrNumInterface
{
public:
	ocrNumInterface() ;
	~ocrNumInterface();
	/*
	 * 题号手写识别接口
	 * IN img --- 待识别的小图
	 * OUT structReg --- 接收识别结果
	 * IN regNumModel ---- 识别模型指针
	 * IN oriLabel  --- 识别的真实内容(标准结果), 默认传空字符串
	 * 返回值：
	 *310 --- 输入的图像为空
	 *311 --- 识别出错
	 *20000 --- 识别成功
	 * */
	int getRegResult(const cv::Mat& img, regResult
					   &structReg, ocrNumRegModel *regNumModel,const std::string& oriLabel = "");
private:
    //  RegNum *regNumModel;	
};
#endif
