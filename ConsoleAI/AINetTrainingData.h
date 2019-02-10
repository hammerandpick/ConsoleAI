#pragma once
#include "CodeFromWeb.h"

class AINetTrainingData
{
public:
	AINetTrainingData();
	~AINetTrainingData();

	
	size_t getTrainingDataRowsMax();
	size_t getTrainingDataColumnsMax();
	size_t getTrainingDataBegin();
	size_t getTrainingDataEnd();
	size_t getTotalLines();
	size_t getNumberOfInputNodes();
	size_t getNumberOfOutputNodes();
	double getTrainingDataValue(size_t column, size_t row);
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
	size_t intMaxIterations = 1000;
	size_t intTimePreviousRows = 0;
	size_t intTimeNextRows = 0;
	size_t intTimePrevNumberOfColumns = 0;
	size_t intTimeNextNumberOfColumns = 0;
	size_t intPercentOfDataToBeUsed = 100;
	size_t intLinesRead = 4;
	std::vector<std::string> vStrTrainingDataColumns = { "no data loaded", "standard-xor" };
	std::vector<size_t> vdNetworkTopology = { 2,2,1 }; // standard xor training data network topology
	std::vector<std::vector<double>> vvTrainingDataMatrix = { {0.0,0.0,1.0},{0.0,1.0,0.0},{1.0,0.0,0.0},{1.0,1.0,0.0} }; // standard xor training data


	/* functions */

	std::vector<size_t> splitStringToSizeT(const std::string & strInput, const std::string & strDelimiter);
	std::vector<double> splitStringToDouble(const std::string & strInput, const std::string & strDelimiter);

	
	std::string convertFromCSVGermanStyle(std::string & strFileContents);
	std::string convertToCSVStandardStyle(std::string & strFileContents);
	

};

