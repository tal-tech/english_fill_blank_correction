#ifndef JUDGEENGLISH_H
#define JUDGEENGLISH_H

#include <string>
#include <vector>
#include "my_util.h"


//英文判题
class judgeEnglish
{
public:
	judgeEnglish();
	~judgeEnglish();	
	
	/*******************************************************************
	判题接口参数：
	IN&OUT vecAllRegPostProcessResult: 识别后处理结果，同时接收判题结果
	返回结果  INT_MAX:2147483647  
	20000       正常
	601  标准答案存在空	
	********************************************************************/	
	static int judgeResult(std::vector<regPostProcessResult>& vecAllRegPostProcessResult);
	
	static char correct(std::string &in_ans, std::string std_ans);
};
#endif

