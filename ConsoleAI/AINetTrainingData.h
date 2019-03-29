#pragma once
#include "CodeFromWeb.h"

class AINetTrainingData
{
public:
	AINetTrainingData();
	~AINetTrainingData();

	
	size_t getTrainingDataRowsMax(bool bReload=false);
	size_t getTrainingDataColumnsMax(bool bRecount=false);
	size_t getTrainingDataBegin();
	size_t getTrainingDataEnd();
	size_t getTotalLines();
	size_t getNumberOfInputNodes();
	size_t getNumberOfOutputNodes();
	size_t getTimeMode();
	double getTrainingDataValue(size_t column, size_t row);
	std::vector<double> getInputDateTime(size_t column);
	size_t getTrainingRowSizeT(size_t row);
	std::vector<size_t> getNetworkTopology();
	std::vector<std::vector<double>> getTrainingDataMatrix();
	std::vector<std::vector<double>>* ptrTrainingDataMatix();
	std::string getTrainingDataFileName();
	std::string setTrainingDataFileName(std::string strFileName);


	bool setOptionCSVGermanStyle(bool bGerStyle);
	bool setPreferredNetworkTopology(std::string strPref);
	bool setPreferredNetworkTopology(std::vector<size_t> vsPref);
	std::string TrainingDataColumnName(size_t tmpColumn, bool shortList=true);

	size_t loadTrainingData(std::string strFileName, bool bSample=false);

	/* move to private section later */
	void closeTrainingDataFile(std::ifstream &ptrDataFile);
	bool openTrainingDataFile(std::ifstream &ptrDataFile);
	size_t loadTrainingDataFile();
	

private:
	
	/* data file variables */
	bool bOptionCSVGER = false;  
	std::string strAIDataFileHeader = "";
	std::string strAIDataFileName = "";

	/* calculation variables */
	size_t intMaxIterations = 1000;
	size_t intTimePreviousRows = 0;
	size_t intTimeNextRows = 0;
	size_t intTimePrevNumberOfColumns = 0;
	size_t intTimeNextNumberOfColumns = 0;
	double dPercentVerificationData = 0.0;
	size_t intLinesRead = 4;
	size_t intTrainingDataColumsMax = 0;
	size_t intTrainingDataRowsMax = 0;
	
	/* data storage */
	std::vector<std::vector<double>> vvTrainingDataMatrix = { {1.0,0.0,0.0,0.0},{1.0,0.0,1.0,1.0},{1.0,1.0,0.0,1.0},{1.0,1.0,1.0,1.0} }; // standard xor training data
	std::vector<std::vector<double>> vvTimingData = { {0.0} }; // this is used for expanding training data to timing data.
	
	/* data information */
	bool bHasTiming = false; // if true timing data has to be considered
	size_t intTimeDataMode = 1; // 1= no specific time data mode; 2=Date; 3=Time; 4=Date and Time
	std::vector<std::string> vStrTrainingDataColumns = { "unused", "Value A", "Value B", "A OR B" };
	std::vector<size_t> vdNetworkTopology = { 2,2,1 }; // standard xor training data network topology
	size_t iNumberOfInputNodes = 2; // this should only be change while loading of training data
	size_t iNumberOfOutputNodes = 1; // this should only be change while loading of training data

	/* functions */
	std::vector<size_t> splitStringToSizeT(const std::string & strInput, const std::string & strDelimiter);
	std::vector<double> splitStringToDouble(const std::string & strInput, const std::string & strDelimiter);
	bool createTimingData();
	
	std::string convertFromCSVGermanStyle(std::string & strFileContents);
	std::string convertToCSVStandardStyle(std::string & strFileContents);
	

};

