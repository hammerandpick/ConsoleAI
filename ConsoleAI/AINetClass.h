#pragma once

#include <memory>
#include "CodeFromWeb.h"
#include "AINetTrainingData.h"


class AINetClass
{
public:
	//variables

	//functions
	AINetClass();
	~AINetClass();

	//AINetClass(const AINetClass &oldClass); 
	size_t NUMNODES();
	size_t NUMINPUTNODES();
	size_t NUMREALINPUTNODES();
	size_t NUMOUTPUTNODES();
	size_t NUMHIDDENNODES();
	size_t SizeOfArray();
	size_t getMaxIterations();
	size_t Counter(bool bIncrease = false);
	size_t CurrentTrainingDataRow();
	size_t getActivationFunction();
	size_t getActivationFunction(size_t tmpNodeID);
	size_t getNumberOfNodesInLayer(signed int iTmpLayer);
	size_t getNumberOfNodesInLayer(size_t tmpLayer);
	size_t getNumberOfLayers(bool bOnlyHidden = false);
	size_t getLayerStart(int iTmpLayer, bool falseForLayerEnd = true);
	
	double LearningRate();
	size_t TrainingDataColumns();
	size_t getTrainingDataRowsMax();	
	size_t getMaximumNodesLayer(bool bGetMaximumNodes = false);
	size_t getLayerByNode(size_t itmpNode);
	bool continueCalculation();
	bool IsTrainingEndOfDataset();
	bool IsLastLayer(int tmpLayer);
	bool linkTrainingDataContainer(std::shared_ptr<AINetTrainingData> ptrToContainer);
	bool getOptionStatus();
	bool setNumInputNodes(size_t tmpInputNodes);
	bool setNumOutputNodes(size_t tmpOutputNodes);
	bool setTimePrevRows(size_t tmpPrevRows);
	bool setTimeInputColumns(size_t tmpPrevCols);
	bool setMaxIterations(size_t tmpMaxIterations);
	bool setLearningRate(double tmpLearningRate);
	bool setNumberOfHiddenLayers(size_t tmpHiddenLayers, size_t tmpNodesinHiddenLayer = 1);
	bool setNumberOfNodesinLayer(int iTmpLayer, size_t tmpNumberOfNodes);
	bool resetCounter();
	bool IsNetworkReady();
	bool autoGenerateInternalNetwork();
	double updateWeights();
	double getNodeValue(size_t tmpNode);
	void TrainingDataColumnPush_Back(std::string tmpString);
	void activateNetwork();
	double calculateErrorMSE(int iTmpLayer);
	void connectNodes(bool bFullyConnected = true, size_t iRandSeed = 0, bool bDeleteExisting = false);
	void combineNetworks(AINetClass *ptrAiNetClass, std::mutex & ptrMutex, size_t iNumber=0);
	void createNetwork(std::vector<std::string> tmpsNetwork);
	void createNetwork(std::vector<size_t> tmpviNetwork);
	void printIO(double sumOfSquaredErrors);
	void displayWeights();
	void displayStatus();
	void displayAllNodes(double sumOfSquaredErrors);
	void initialize(std::vector<size_t> iInternalTopology = { 0 });
	void saveResultingNetwork(size_t iNumber = 0);
	void setActivationFunction(size_t typeOfActivationFunction, size_t specificLayer = 0);
	void setDataFileName(std::string strFileName);
	void setInternalName(std::string strIntName);
	void setOptionAutoGenerate(bool bAutoGenerate);
	void setOptionCSV(bool bSetGerman);
	void setOptionDisplayAllNodes(bool bDisplayAll);
	void setOptionIO(bool bSetIO);
	void setOptionNoDeep(bool bSetNoDeep);
	void setOptionShuffle(bool bSetShuffle);
	void setOptionSilent(bool bSilent);
	void setOptionStatus(bool bSetStatus);
	void setOptionWeight(bool bSetWeight);
	void setOptionNodeFunction(size_t tmpNodeFunction);
	void setOptionThreadCombinatingMode(size_t iTCMode);
	void setPercentVerification(double tmpPercentVerifiy);
	void setTrainingRow(size_t iTmpRow);
	void calculateLine(size_t iTmpRow);
	
	void sortNetwork();

	/* Data File */
	std::string getDataFileName();
	std::string getAIDataFileHeader();


	/* Training Data */
	std::vector<std::vector<double>> *getTrainingData();
	void shuffleTrainingData();
	std::string getTrainingDataOptionTimeString();

	
	void trainNetwork(bool bSilent=false);
	
	std::vector<std::string> getErrorList();
	std::vector<std::string> splitString(const std::string& strInput, const std::string& strDelimiter);

private:
	// Constants

	double E = 2.71828;

	// variables
	std::vector<double> vecValues = { 0.0 }; // vector containing values including input
	std::vector<double> vecExpectedValues = { 0.0 }; // vector containing training outputvalues
	std::vector<double> vecThresholds = { 0.0 };; // vector containing threshold values,Theshold also known as bias. used as automatic linear offset in calculation
	std::vector<std::vector<double>> vecWeights = { { 0.0 } }; // matrix containing weight between nodes
	std::vector<double> vecCalcDelta = { 0.0 };
	std::vector<size_t> vdNetworkTopology = { 2,2,1 }; // standard xor training data network topology


	/* Data File */
	std::string strAIDataFileName = "";
	std::string strAIDataFileHeader = "";
	
	bool bHasBeenConnected = false;
	bool bOptionShuffle = false;
	bool optionWeight = false;
	bool bOptionStatus = false;
	bool bOptionIO = false;
	bool bSilent = false;
	bool optionNoDeep = false;
	bool bOptionDisplayAllNodes = false;
	bool bOptionAutoGenerate = false;
	bool bOptionCSVGER = false;
	bool bOptionMaxIterationSet = false;
	bool initializationDone = false;
	bool bHistoricData = false;
	bool bFutureData = false;
	bool bBackpropagationActive = false;
	double dPercentVerification = 0.0;	// set the percentage of verification data
	size_t iThreadedCombinationMode = 0; // set the mode for combining data
	size_t iActivationFunction = 0;
	size_t iNumInputNodes = 2;	// basic number of input nodes
	size_t iNumRealInputNodes = 2;
	size_t iNumOutputNodes = 1;	// number of output nodes
	size_t iTimePreviousRows = 0;	// number of previous rows for time-dependent calculation
	size_t iTimeNumInputColumns = 0;	// number of columns for time-dependent calculation
	size_t iTimeNextRows = 0;
	size_t iTimeNumOutputColumns = 0;
	size_t iMaxIterations = 1000;
	size_t iCounter = 0;
	size_t iTrainingDataRow = 0;
	double dlearningRate = 0.2; // the default learning rate
	double dWorstError = 0.0;
	size_t iWorstErrorRow = 0;
	std::string strInternalName = "Sample Data for A OR B";
	std::vector<std::string> vTrainingDataColumns={ "0" };
	std::vector<size_t> inputDataPullList = { 0 };
	std::vector<size_t> viLayerActivationFunction = { 0 };
	std::vector<std::string> errorList = { "0" };
	std::vector<std::vector<double>> vvErrors={ {0.0} };
	std::vector<std::string> vstrResultFilenames = { "" };
	std::shared_ptr<AINetTrainingData> ptrAINDataContainer = nullptr;
	
	// functions
	std::string generateFileOutput(std::string& strFileContents);
	std::string generateFileInput(std::string& strFileContents);

	bool IsDoubleCritical(double dToBeClassified);
	std::string IsDoubleCritical(double dToBeClassified, std::string sText);
	bool recalculateInputDataPullList();
	bool throwFailure(std::string tmpError, bool doexit);
	double getTrainingDataValue(size_t row, size_t column);
	double updateWeightsInLayer(signed int iTmpLayer);
	std::string getExcelColumn(size_t stColumn);
	size_t getNumberOfInputNodes();
	size_t getNumberOfOutputNodes();
	size_t resizeVectors();
	size_t validLayer(signed int tmpLayer);
	size_t validLayer(size_t& tmpLayer);

	void loadTrainingLine();
	double NodeFunction(double weightedInput, size_t currentNodeID, bool derivative = false);
	std::string NodeFunctionXLS(size_t tmpNode, std::string tmpCalculatedInput);
	std::string NodeFunctionJS(size_t tmpNode, std::string tmpCalculatedInput);

};
