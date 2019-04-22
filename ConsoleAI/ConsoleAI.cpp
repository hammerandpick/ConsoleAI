// ConsoleAI.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <Windows.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <climits>
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <memory>
#include <time.h>
#include <mutex>
#include "AINetClass.h"


// variables
size_t iTimeNumInputColumns = 0;	// How many columns for previous calculation?
size_t iTimeNextElements = 0;
size_t iTimeNumOutputColumns = 0;
const int iThreads = 0;	// number of calculation threads
bool bTrainingDataRowsCounting = true;
double MOMENTUM = 0.0;
size_t theAIDataFilePos = 0;
char theAIWeightsFileName[] = "weights.aiweights.csv";
char *cThisFileName;
const double VERSION = 0.20190221;
bool optionsAuto = false;
bool optionsWeightSave = false;
bool optionsAllNodes = false;
bool optionMaxIterationsSet = false;
bool optionsNoDeep = false; // turning off deep network

auto ainTrainingData = std::make_shared<AINetTrainingData>();
AINetClass aincNetwork; // dataContainer for the Network
std::mutex myMutex; // for multithreading

int main(int, char*[], char*[]);

// Functon prototypes
bool modifyInputs(AINetClass*, std::string);
void pause();
void printTrainingData(AINetClass*);
void threadedCalculation(AINetClass *ptrToNetwork, size_t iThreadID);

// pre-exit function
void leaveApplication();


/*******************************
/
/	begin main application
/
*******************************/

int main(int argc, char *argv[], char *env[]) {
	/** This is the main function of the application.
	*/
	// Welcome Screen
	printf("Neural Network Program\n(%s)\n", argv[0]);
	printf("Version:%10.8f \n\n--- OPTIONS ---", VERSION);

	atexit(leaveApplication);

	// environment variable
	::cThisFileName = argv[0];

	// variables 
	int chooseMode = 0;
	int theFirstElement = 0;

	// checking parameters for programm
	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?") || !strcmp(argv[i], "/?") || !strcmp(argv[i], "-help") || !strcmp(argv[i], "/h") || !strcmp(argv[i], "/help"))
		{
			printf("Usage:\n%s [options] [filename]\n\n", argv[0]);
			printf("List of options:\n-a\tshow all nodes\n");
			printf("-auto\t non-interactive-mode\n");
			printf("-autonet\t auto-generate internal network\n");
			printf("-csvger\t csv german style (0,0;0,0)\n");
			printf("-file\t [-file filename]\tuse specified file\n");
			printf("-func-bip\t use bipolar sigmoid  activation function\n");
			printf("-func-tanh\t use tanh activation function\n");
			printf("-func-lin\t use linear activation on output layer function\n");
			printf("-func-mix\t use tanh activation within and sigmoid out function\n");
			printf("-func-relu\t use relu on inner and linear on outer node\n");
			printf("-io\tshow io data while calculating\n");
			printf("-learn\t [-learn number]\tset learning rate\n");
			printf("-nodeep\tturning off deep nodes\n");
			printf("-h\tshow (this) help screen and exit\n-s\tshow status\n-save\tsave the weights to a file\n");
			printf("-maxit\t [-iteration number]\tset maxiteration\n");
			printf("-shuffle\tshuffle data after each try\n");
			printf("-w\tshow weights when finished\n");
			exit(0);
		}
		else if (!strcmp(argv[i], "-auto")) { printf("using non-interactive-mode\n"); ::optionsAuto = true; }
		else if (!strcmp(argv[i], "-autonet")) { printf("auto-generate internal network\n"); ::optionsAuto = true; }
		else if (!strcmp(argv[i], "-s")) { printf("display Status enabled\n"); aincNetwork.setOptionStatus(true); }
		else if (!strcmp(argv[i], "-w")) { printf("display Weights enabled\n"); aincNetwork.setOptionWeight(true); }
		else if (!strcmp(argv[i], "-a")) { printf("display all nodes enabled\n"); ::optionsAllNodes = true; }
		else if (!strcmp(argv[i], "-csvger")) { printf("csv in german style\n"); aincNetwork.setOptionCSV(true);  ainTrainingData->setOptionCSVGermanStyle(true); }
		else if (!strcmp(argv[i], "-io")) { printf("display I/O-Data enabled\n"); aincNetwork.setOptionIO(true); }
		else if (!strcmp(argv[i], "-nodeep")) { printf("deep network disabled.  ATTENTION output function may not work properly.\n"); aincNetwork.setOptionNoDeep(true); }
		else if (!strcmp(argv[i], "-save")) { printf("save Weights enabled\n"); ::optionsWeightSave = true; }
		else if (!strcmp(argv[i], "-func-tanh")) { printf("activation function set to tanh\n"); aincNetwork.setOptionNodeFunction(1); }
		else if (!strcmp(argv[i], "-func-mix")) { printf("activation function set to tanh for inner nodes\n"); aincNetwork.setOptionNodeFunction(5); }
		else if (!strcmp(argv[i], "-func-lin")) { printf("activation function set to linear for output nodes\n"); aincNetwork.setOptionNodeFunction(4); }
		else if (!strcmp(argv[i], "-func-bip")) { printf("activation function set to bipolardigmoid\n"); aincNetwork.setOptionNodeFunction(2); }
		else if (!strcmp(argv[i], "-func-relu")) { printf("activation function set to relu for inner nodes and linear for output nodes\n"); aincNetwork.setOptionNodeFunction(3); }
		else if (!strcmp(argv[i], "-file")) { printf("using file %s\n", argv[min(i + 1, argc - 1)]); aincNetwork.setDataFileName(argv[min(i + 1, argc - 1)]); }
		else if (!strcmp(argv[i], "-maxit")) { printf("setting maximum iterations to %i\n", atoi(argv[min(i + 1, argc - 1)])); optionMaxIterationsSet = true; aincNetwork.setMaxIterations(atoi(argv[min(i + 1, argc - 1)])); }
		else if (!strcmp(argv[i], "-learn")) { printf("setting learning rate to %f\n", atof(argv[min(i + 1, argc - 1)])); aincNetwork.setLearningRate(atof(argv[min(i + 1, argc - 1)])); }
		else if (!strcmp(argv[i], "-shuffle")) { printf("shuffling activated\n"); aincNetwork.setOptionShuffle(true); }
		else {
			//dismiss all other ones
		}
	}
	printf("--- OPTIONS END ---");

	if (!::optionsAuto)
	{
		// asking if standard or file mode
		printf("0 = Standard Mode (internal Data)\n");
		printf("1 = File Mode (Load %s)\n", aincNetwork.getDataFileName().c_str());
		printf("2 = Open File...\n");
		printf("Please select mode:");
		while (!(std::cin >> chooseMode))
		{
			std::cout << "only Numbers!" << std::endl;
			std::cin.clear();
			std::cin.ignore(std::cin.rdbuf()->in_avail());
		}
	}


	aincNetwork.linkTrainingDataContainer(ainTrainingData);

	size_t iFileErrors = 0;
	std::string theLine;
	if (((chooseMode == 1) || (::optionsAuto)) && (0 != strcmp("", aincNetwork.getDataFileName().c_str())))
	{
		iFileErrors = ainTrainingData->loadTrainingData(aincNetwork.getDataFileName());
	}
	else if (chooseMode == 2)
	{
		std::string strCurrentDir = argv[0];
		std::string strFile = "";
		strCurrentDir = strCurrentDir.substr(0, strCurrentDir.find_last_of("\\")+1);
		printf("enter filename: \t%s",strCurrentDir.c_str());
		std::cin >> strFile;
		strFile = strCurrentDir + strFile;
		printf("loading %s...", strFile.c_str());
		pause();
		iFileErrors = ainTrainingData->loadTrainingData(strFile);
	}
	else if (chooseMode == 3)
	{
		/** TODO: REMOVE - THIS IS ONLY USED FOR DEBUG */
		printf("enter number of file:\t\n");
		std::string strCurrentDir = aincNetwork.getDataFileName();
		std::string strFile = "";
		strCurrentDir = strCurrentDir.substr(0, strCurrentDir.find_last_of("\\") + 1);
		size_t intChooseFile = 0;
		while (!(std::cin >> intChooseFile))
		{
			std::cout << "only Numbers!" << std::endl;
			std::cin.clear();
			std::cin.ignore(std::cin.rdbuf()->in_avail());
		}
		switch (intChooseFile)
		{
		case 1:
			strFile = strCurrentDir + "training.aidata.csv";
			break;
		case 2:
			strFile = strCurrentDir + "training-weather.aidata.csv";
			break;
		default:
			strFile = aincNetwork.getDataFileName();
			break;
		}
		printf("file %s selected.\n", strFile.c_str());
		aincNetwork.setDataFileName(strFile);
		iFileErrors = ainTrainingData->loadTrainingData(strFile);
	}
	else
	{
		iFileErrors = ainTrainingData->loadTrainingData((std::string)"Sample",true);
	}

	printf("Read training data file.\n Read %zu lines of which %zu were erroneus.\n", ainTrainingData->getTotalLines(), iFileErrors);
	printf("Timing mode:\t%s\n", aincNetwork.getTrainingDataOptionTimeString().c_str());

	aincNetwork.initialize();
	aincNetwork.connectNodes();
	aincNetwork.displayStatus();
	aincNetwork.setOptionThreadCombinatingMode(0);
	printTrainingData(&aincNetwork);
	pause();

	printf("\nnumber of threads: %i", iThreads);
	std::string sEnteredData = "";
	while (modifyInputs(&aincNetwork, sEnteredData) && !optionsAuto)
	{
		std::cin >> sEnteredData;
	}
	return 0;
}

bool modifyInputs(AINetClass* ptrAINetClass, std::string sEnterData)
{
	bool bRunAgain = true;
	std::transform(sEnterData.begin(), sEnterData.end(), sEnterData.begin(), ::tolower);
	if (sEnterData == "q" || sEnterData == "quit" || sEnterData == "exit") { bRunAgain = false; }
	if (sEnterData == "status") { ptrAINetClass->setOptionStatus(!ptrAINetClass->getOptionStatus()); ptrAINetClass->displayStatus(); }	// now toggling status and displaying data
	if (sEnterData == "io") { ptrAINetClass->setOptionIO(true); ptrAINetClass->printIO(0.0); } // if options set displaying all net
	if (sEnterData == "node") { ptrAINetClass->displayAllNodes(0.0); } // if options ist set displaying all nodes
	if (sEnterData == "savenet") { ptrAINetClass->saveResultingNetwork(); }
	if (sEnterData == "p" || sEnterData == "print") { printTrainingData(ptrAINetClass); }
	if (sEnterData == "act-tanh") { ptrAINetClass->setActivationFunction(1); }
	if (sEnterData == "act") { ptrAINetClass->setActivationFunction(0); }
	if (sEnterData == "reset") { ptrAINetClass->resetCounter(); ptrAINetClass->connectNodes(true, 0, true); }
	if (sEnterData == "cn" || sEnterData == "calc" || sEnterData == "calculate")
	{
		ptrAINetClass->resetCounter();
		// calculate the network
		if (iThreads > 0)
		{
			// initializing thread variables
			std::vector<std::thread> vtThread;
			vtThread.clear();
			//This statement will launch multiple threads in loop
			for (int i = 0; i < iThreads; ++i) {
				//start thread
				vtThread.push_back(std::thread(threadedCalculation, ptrAINetClass, i));
			}
			for (auto& thread : vtThread) {
				// TODO: exception thrown when using internal data.
				thread.join();
			}
			// get data from network 
		}
		else {
			ptrAINetClass->connectNodes(true, 0);
			ptrAINetClass->trainNetwork(false);
		}
	}
	if (!strcmp(sEnterData.c_str(), "switch"))
	{
		// now check if data has been specified.
		std::cin >> sEnterData;
		size_t iDataset = atoi(sEnterData.c_str());
		if ((iDataset > 0) && (iDataset <= ptrAINetClass->getTrainingDataRowsMax()))
		{
			ptrAINetClass->setTrainingRow(iDataset);
		}
	}
	if (!strcmp(sEnterData.c_str(), "cl"))
	{
		// now check if data has been specified.
		std::cin >> sEnterData;
		size_t iDataset = atoi(sEnterData.c_str());
		if ((iDataset > 0) && (iDataset <= ptrAINetClass->getTrainingDataRowsMax()))
		{
			ptrAINetClass->calculateLine(iDataset);
		}
	}
	else
	{
		// todo current values
	}
	printf("\nEnter>");
	return bRunAgain;
}

void pause()
{
	// pause the application
	if (!::optionsAuto)
	{
		std::cout << "\nPress ENTER to continue..." << std::endl;
		std::cin.ignore(10, '\n');
		std::cin.get();
	}
}

void printTrainingData(AINetClass *ptrAINC)
{
	/** This function is used to print training data to standard output. */

	const std::vector<std::vector<double>> *vvTmpTrainingData = ptrAINC->getTrainingData();
	std::vector<double> vTrainingDataRow;

	printf("\n");

	for (size_t iTrainingDataRow = 0; iTrainingDataRow < vvTmpTrainingData->size(); iTrainingDataRow++)
	{
		// do it row by row, copy row to temp var.
		vTrainingDataRow = vvTmpTrainingData->at(iTrainingDataRow);
		printf("\n");
		for (size_t iTrainingDataColumn = 0; iTrainingDataColumn < vTrainingDataRow.size(); iTrainingDataColumn++)
		{
			printf("%4f|", vTrainingDataRow.at(iTrainingDataColumn));
		}
	}
}

void threadedCalculation(AINetClass *ptrToNetwork, size_t iThreadID)
{
	// calculation in threaded mode
	ptrToNetwork->setInternalName("Network_" + std::to_string(iThreadID));
	ptrToNetwork->connectNodes(true, iThreadID);
	if (iThreadID > 0)
	{
		ptrToNetwork->setOptionSilent(true);
		ptrToNetwork->setLearningRate((1.0 / iThreadID));
	}
	// training
	ptrToNetwork->trainNetwork();
	std::cout << "\nCalculation Thread " << iThreadID << " finished.";
	aincNetwork.combineNetworks(ptrToNetwork, ::myMutex, iThreadID);
}

void leaveApplication()
{
	/** This function is being called on exit of the application. It will print the internal error messages on the screen.
	*/
	std::vector<std::string> tmpErrorList;
	std::vector<std::string> tmpNetworkErrorList;
	bool noErrors = true;
	tmpNetworkErrorList = aincNetwork.getErrorList();
	tmpErrorList.clear();
	tmpErrorList.insert(tmpErrorList.end(), tmpNetworkErrorList.begin(), tmpNetworkErrorList.end());
	printf("\n\nnow leaving application. %zu errors have appeared.", tmpNetworkErrorList.size());
	if (tmpErrorList.size() > 0)
	{
		for (size_t i = 0; tmpErrorList.size(); i++)
		{
			printf("\nError %8zu\t%s", i, tmpErrorList.at(i).c_str());
		}
		pause();	// show to user
	}
	printf("\nbye.");
}
