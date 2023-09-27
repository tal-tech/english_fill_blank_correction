#ifndef REGPOSTPROCESS_H
#define REGPOSTROCESS_H
#include "my_util.h"
//识别后处理类
class regPostProcess
{
public:
	regPostProcess();
	/*
	识别后处理接口，匹配题号识别结果与手写识别结果，对涂改结果修正
	参数：
	IN vecAllRegResultHW  --- 英文手写识别结果结构
	IN vecAllRegResultNO --- 题号识别结果结构
	IN detPostProcessResult --- 检测后处理结果
	IN strAns  --- 标准答案
	OUT vecAllRegPostProcessResult  --- 接收识别后处理结果
    返回：
	501 ---  识别结果为空
	502 --- 识别后处理错误
	503 --- 依据检测/识别的题目划分错误
	504 --- 依据检测/识别的题目划分数量错误
	505 --- 标准答案格式错误
	506 --- 标准答案为空
	20000  --- 处理正常无错误
	*/
	long getRegPostProcessResult(const std::vector<regResult>& vecAllRegResultHW, const std::vector<regResult>& vecAllRegResultNO,const std::vector<detPostProcessResult>& vecAllDetPostResult, const std::string& strAns, std::vector<regPostProcessResult>& vecAllRegPostProcessResult);


	
	virtual ~regPostProcess();
private:
	/*
	从答案json加载标准答案字段
	参数：
	IN strAnsJson --- 标准答案，json格式
	OUT vecAllAnswer --- 接收json数据解析出来的各字段
	返回值：
	505 --- 标准答案格式错误
	506 --- 标准答案为空
	20000  --- 处理正常无错误
	*/
	long loadAnswer(const std::string& strAnsJson, std::vector<answer>& vecAllAnswer);

	/*
	从检测后处理结果和识别结果初始化识别后处理数据
	IN vecAllRegResultHW  --- 英文手写识别结果结构
	IN vecAllRegResultNO --- 题号识别结果结构
	IN detPostProcessResult --- 检测后处理结果
	OUT vecAllRegPostProcessData  --- 接收初始化后处理结果
	返回值：
	501 --- 识别后处理输入的识别结果为空
	503 --- 识别和检测结果不匹配
	20000  --- 处理正常无错误
	*/
	long initRegPostProcessData(const std::vector<regResult>& vecAllRegResultHW, const std::vector<regResult>& vecAllRegResultNO, const std::vector<detPostProcessResult>& vecAllDetPostResult, std::vector<regPostProcessResult>& vecAllRegPostProcessData);
};

#endif
