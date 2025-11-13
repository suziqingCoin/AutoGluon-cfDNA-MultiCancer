1 .  使用随机森林和mRMR进行特征选择: http://blog.csdn.net/wtq1993/article/details/51540321
     随机森林和mRMR特征选择:http://www.tuicool.com/articles/FBRF7bq
     使用举例：https://github.com/csuldw/MachineLearning/tree/master/mRMR 
	 combine_f195.data -t 0.0001 -s 1000000 -v 1000 -n 42 > result.out  
	 
	 Usage: mrmr_osx -i <dataset> -t <threshold> [optional arguments]
         -i <dataset>    .CSV file containing M rows and N columns, row - sample, column - variable/attribute.
         -t <threshold> a float number of the discretization threshold#离散化阈值; non-spec ifying this parameter means no discretizaton (i.e. data is already integer); 0 to make binarization.
		 ########     threshold=mean+/-t*std      #######
         -n <number of features>   a natural number, default is 50.
         -m <selection method>    either "MID" or "MIQ" (Capital case), default is MID.
         -s <MAX number of samples>   a natural number, default is 1000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
         -v <MAX number of variables/attibutes in data>   a natural number, default is 10000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
         [-h] print this message.
	 
2. 【特征工程】特征选择及mRMR算法解析:http://www.jianshu.com/p/04d965e35b6d


命令1:mrmr_win32.exe -i 0.1_24_4_n_CpG.csv -t 0.001 -n 402 -s 5387 -v 402 > 1ResultMRMR.txt

命令2:mrmr_win32.exe -i 0.1_24_4_n_CpG.csv -t 0 -n 402 -m MIQ -s 5387 -v 402 > 2ResultMRMR.txt

命令3:mrmr_win32.exe -i 423.csv -t 0 -n 423 -s 5387 -v 423 > 423ResultMRMR.txt
