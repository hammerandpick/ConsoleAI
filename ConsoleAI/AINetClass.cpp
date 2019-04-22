/*

	AINetClass

	This class is used to generate a network an perform calculation on this network.

*/

#include "pch.h"
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <time.h>
#include <mutex>
#include <memory>
#include "CodeFromWeb.h"
#include "AINetTrainingData.h"
#include "AINetClass.h"


AINetClass::AINetClass()
{
	this->iNumInputNodes = 2;
	this->iNumOutputNodes = 1;
	this->ptrAINDataContainer = nullptr;
	this->iMaxIterations = 1000;
	this->dlearningRate = 0.2;
	this->iCounter = 0;
	this->iActivationFunction = 0;
	this->bOptionShuffle = false;
	this->bOptionIO = false;
	this->vTrainingDataColumns.resize(this->iNumInputNodes);
	this->vTrainingDataColumns.clear();
	this->vecValues.clear();
	this->vecWeights.clear();
	this->vecThresholds.clear();
	this->vecExpectedValues.clear();
	this->vecCalcDelta.clear();
	this->errorList.clear();
	this->vvErrors.clear();
	this->iTimeNumInputColumns = 0;
	this->iTimePreviousRows = 0;
	this->bOptionMaxIterationSet = false;
	this->bOptionAutoGenerate = false;
	this->bOptionDisplayAllNodes = false;
}

AINetClass::~AINetClass()
{
	this->vecExpectedValues.clear();
	this->vecValues.clear();
	this->vecWeights.clear();
	this->vecThresholds.clear();
	this->vecCalcDelta.clear();
}

size_t AINetClass::NUMNODES()
{
	/** This function returns the number of nodes of the network.
		\return number of nodes.
	*/
	size_t totalNumberNodes = 0;
	for (size_t currentLayer = 0; currentLayer < this->vdNetworkTopology.size(); ++currentLayer)
	{
		totalNumberNodes += this->vdNetworkTopology.at(currentLayer);
	}
	return totalNumberNodes;
}

size_t AINetClass::NUMINPUTNODES()
{
	/** This is a wrapper function.
	*/
	return this->getNumberOfInputNodes();
}

size_t AINetClass::NUMREALINPUTNODES()
{
	/** This is a wrapper function.
	*/
	return this->iNumRealInputNodes;
}

size_t AINetClass::NUMOUTPUTNODES()
{
	/** This is a wrapper function.
	*/
	return this->getNumberOfOutputNodes();
}

size_t AINetClass::NUMHIDDENNODES()
{
	/** This function returns the number of hidden nodes in the network
		\return number of hidden nodes as size_t
	*/
	// return number of hidden nodes
	size_t iCalc = 0;
	if (this->vdNetworkTopology.size() > 2)
	{
		// no internal nodes
		for (size_t i = 1; i < this->vdNetworkTopology.size() - 1; ++i)
		{
			iCalc += this->vdNetworkTopology.at(i);
		}
	}
	return iCalc;
}

size_t AINetClass::SizeOfArray()
{
	return 1+this->NUMNODES();
}

size_t AINetClass::getMaxIterations()
{
	// returns the number of maxIterations
	return iMaxIterations;
}

size_t AINetClass::Counter(bool bIncrease )
{
	if (bIncrease)
	{
		this->iCounter += 1;
		// now shuffle the list
		if ((this->iCounter % this->getTrainingDataRowsMax() == 0) && bOptionShuffle)
		{
			this->shuffleTrainingData();
		}
	}
	return this->iCounter;
}

size_t AINetClass::CurrentTrainingDataRow()
{
	// returns current Training Data Row
	size_t tmpReturn = 0;
	size_t tmpMaxRows = this->getTrainingDataRowsMax();
	if (tmpMaxRows == 0)
	{
		this->throwFailure("division by 0 iTrainingDataRowsUseMax", true);
	}
	else
	{
		size_t iDiv = this->iCounter % tmpMaxRows;
		// pull the date from the list.
		if (iDiv < this->inputDataPullList.size())
		{
			tmpReturn = this->inputDataPullList.at(iDiv);
		}
		else
		{
			this->throwFailure("pull list exeeded.", true);
		}
	}
	return tmpReturn;
}

size_t AINetClass::getActivationFunction()
{
	return this->iActivationFunction;
}

size_t AINetClass::getActivationFunction(size_t tmpNodeID)
{
	// this function returns the correct activation function for requested node
	size_t tmpLayer = this->getLayerByNode(tmpNodeID);
	return this->viLayerActivationFunction.at(min(tmpNodeID, this->viLayerActivationFunction.size() - 1));
}

size_t AINetClass::getNumberOfNodesInLayer(signed int iTmpLayer)
{
	/** This will return the number of nodes in one layer, see also getNumberOfNodesInLayer(size_t tmpLayer, bool fromEnd)
		\param iTmpLayer (signed int) layer in question. if negative it will be counted from output layer.
		\return (size_t) will return the number of the layer
	*/
	size_t tmpReturn = 0;

	size_t tmpLayer = this->validLayer(iTmpLayer);
	tmpReturn = this->vdNetworkTopology.at(max(0, tmpLayer - 1));
	return tmpReturn;
}

size_t AINetClass::getNumberOfNodesInLayer(size_t tmpLayer)
{
	/** This will return the number of nodes in one layer, see also getNumberOfNodesInLayer(signed int tmpLayer, bool fromEnd)
		\param tmpLayer (signed int) layer in question. 
		\return (size_t) will return the number of the layer
	*/

	size_t tmpReturn = 0;

	this->validLayer(tmpLayer);
	tmpReturn = this->vdNetworkTopology.at(max(0, tmpLayer - 1));
	return tmpReturn;
}

size_t AINetClass::getNumberOfLayers(bool bOnlyHidden)
{
	// returns the number of layers
	if (bOnlyHidden) return(size_t) this->vdNetworkTopology.size()-2;// removing input and output layer
	else return (size_t) this->vdNetworkTopology.size();
}

size_t AINetClass::getLayerStart(int tmpLayer, bool falseForLayerEnd)
{
	/** Call for number of first node inspecified layer
		\param tmpLayer Specifiy layer.
		\param falseForLayerEnd (optional) If set to false this will return the last node of the layer.
		\return Returnvalue is number of node at begin (or end) of specified layer. minimum of returnvalue is 1.
	*/
	
	size_t chosenLayer = 0;
	size_t retInt = 0;
	chosenLayer = this->validLayer(tmpLayer) - 1;

	if (chosenLayer == 0 && falseForLayerEnd)
	{
		retInt = 1;
	}
	else
	{
		for (size_t i = 0; i <= chosenLayer; ++i)
		{
			if (i < chosenLayer)
			{
				// sum all previous layers
				retInt += this->vdNetworkTopology.at(i);
			}
			else
			{
				// now adding up the last (the choosenLayer
				if (falseForLayerEnd)
				{
					retInt += 1; // add one for begin of layer
				}
				else
				{
					// add the number of elements in the layer 
					retInt += this->vdNetworkTopology.at(i);
				}
				break;
			}
		}
	}
	return max((size_t)1, retInt);
}

double AINetClass::LearningRate()
{
	// returns current/selected learning rate
	return dlearningRate;
}

size_t AINetClass::TrainingDataColumns()
{
	// this one returns the number of columns filled with names
	return vTrainingDataColumns.size();
}

size_t AINetClass::getTrainingDataRowsMax()
{
	//returns number of training data rows
	/* this one calculates the number of training data columns
	/ which is total number of rows reduced by
	/	- previous Data (historical)
	/	- next Data (historical)
	/	- portion of verification data
	*/
	return max(1,this->ptrAINDataContainer->getTrainingDataRowsMax() - this->iTimePreviousRows);
}

size_t AINetClass::getMaximumNodesLayer(bool bGetMaximumNodes)
{
	// returns number of layer layer with max nodes or maximum number of nodes in a layer (if true is set)
	size_t retValue = this->NUMINPUTNODES();
	size_t retLayer = 0;
	for (size_t currentLayer = 0; currentLayer < this->getNumberOfLayers();currentLayer++)
	{
		retValue = max(retValue, this->vdNetworkTopology.at(currentLayer));
		if (retValue == this->vdNetworkTopology.at(currentLayer))
		{
			retLayer = currentLayer+1;
		}
	}
	
	if (bGetMaximumNodes)
		return retValue;
	else
		return retLayer;
}

size_t AINetClass::getLayerByNode(size_t iTmpNode)
{
	// this function returns the corresponding layer for a specific node
	size_t returnLayer = 0;
	size_t LayerBegin = 1;
	for (size_t i = 0; i < this->vdNetworkTopology.size(); i++)
	{
		if ((LayerBegin <= iTmpNode) && (iTmpNode < LayerBegin + this->vdNetworkTopology.at(i)))
		{
			// this is the correct layer
			returnLayer = i;
			break;
		}
		else
		{
			LayerBegin += this->vdNetworkTopology.at(i);
		}
	}
	return returnLayer;
}

bool AINetClass::continueCalculation()
{
	/** This function is used to check if the calculation should be continued. This is the case if the counter has not reached maxiterations
		\return true if calculation should be continued, otherwise false.
		*/
	
	bool bContCalc = false;
	if (this->Counter() < this->getMaxIterations())
	{
		bContCalc=true;
	}
	return bContCalc;
}

bool AINetClass::IsTrainingEndOfDataset()
{
	/** This function is used to check if the end of the dataset has been reached.
		\return True if end has been reached, otherwise false.
	*/
	if (iCounter % this->getTrainingDataRowsMax() == 0)
		return true;
	else 
		return false;
}

bool AINetClass::IsLastLayer(int tmpLayer)
{
	/** This function can be used to check if a specified layer is the output layer.
		\param tmpLayer The layer in question.
		\return True if it is the output layer, otherwise false.
	*/
	if (tmpLayer == this->vdNetworkTopology.size())
		return true;
	else
		return false;
}

bool AINetClass::linkTrainingDataContainer(std::shared_ptr<AINetTrainingData> ptrToContainer)
{
	/** This function is used to link the training data class to this network
		\param ptrToContainer this is a shared pointer to a container of training data.
		\return always true.
	*/
	this->ptrAINDataContainer = std::shared_ptr<AINetTrainingData>(ptrToContainer);
	return true;
}

bool AINetClass::getOptionStatus()
{
	// returns option status
	return this->bOptionStatus;
}

bool AINetClass::setNumInputNodes(size_t tmpInputNodes)
{
	// set the number of input nodes
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iNumInputNodes = min(max(1, tmpInputNodes), UINT_MAX);
	this->vdNetworkTopology.at(0) = this->iNumInputNodes;
	this->iNumRealInputNodes = this->iNumInputNodes;
	this->resizeVectors();
	return (iNumInputNodes == tmpInputNodes);
}

bool AINetClass::setNumOutputNodes(size_t tmpOutputNodes)
{
	// set the number of output nodes
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iNumOutputNodes = min(max(1, tmpOutputNodes), UINT_MAX);
	this->vdNetworkTopology.at(this->vdNetworkTopology.size() - 1) = this->iNumOutputNodes;
	return (iNumOutputNodes == tmpOutputNodes);
}

bool AINetClass::setTimePrevRows(size_t tmpPrevRows)
{
	/** This function is used to set the number of prevoius rows to take into account.
		\param tmpPrevRows must be positive
	*/
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	this->iTimePreviousRows = tmpPrevRows;
	if (tmpPrevRows == 0)
	{
		bHistoricData = false;
	}
	else
	{
		bHistoricData = true;
		if (this->iTimeNumInputColumns == 0)
		{
			this->iNumInputNodes = this->iNumRealInputNodes * (1 + this->iTimePreviousRows);
		}
		else
		{
			this->iNumInputNodes = this->iNumRealInputNodes + this->iTimeNumInputColumns * this->iTimePreviousRows;
		}
	}
	this->resizeVectors();
	this->recalculateInputDataPullList();
	return (iTimePreviousRows == tmpPrevRows);
}

bool AINetClass::setTimeInputColumns(size_t tmpPrevCols)
{
	// set the number of input columns
	this->initializationDone = false;		// significant changes to network, so initialization should be renewed
	if (this->iTimePreviousRows == 0)
	{
		bHistoricData = false;
	}
	else
	{
		bHistoricData = true;
		this->iNumInputNodes = this->iNumRealInputNodes + this->iTimePreviousRows * tmpPrevCols;
	}
	this->iTimeNumInputColumns = min(max(0, tmpPrevCols), this->iNumRealInputNodes);
	this->resizeVectors();
	return (iTimeNumInputColumns == tmpPrevCols);
}

bool AINetClass::setMaxIterations(size_t tmpMaxIterations)
{
	// SetMaximumNumber of iterations
	this->bOptionMaxIterationSet = true;
	this->iMaxIterations = min(max(1, tmpMaxIterations), UINT_MAX);
	return (this->iMaxIterations == tmpMaxIterations);
}

bool AINetClass::setLearningRate(double tmpLearningRate)
{
	// set the learning rate
	dlearningRate = min(max(0,tmpLearningRate),500);
	return (dlearningRate == tmpLearningRate);
}

bool AINetClass::setNumberOfHiddenLayers(size_t tmpHiddenLayers, size_t tmpNodesinHiddenLayer)
{
	// set the number of hidden layers
	// initialization required
	this->initializationDone = false;
	if (this->optionNoDeep)
	{
		tmpHiddenLayers = 1;
	}
	this->vdNetworkTopology.resize(max(2, tmpHiddenLayers + 2), max(1,tmpNodesinHiddenLayer));
	this->viLayerActivationFunction.resize(max(2, tmpHiddenLayers + 2), 0);
	// set number of input nodes
	this->vdNetworkTopology.at(0) = this->iNumInputNodes;
	// set number of output nodes
	this->vdNetworkTopology.at(this->vdNetworkTopology.size() - 1) = this->iNumOutputNodes;
	return (this->vdNetworkTopology.capacity() == tmpHiddenLayers + 2);
}

bool AINetClass::setNumberOfNodesinLayer(int iTmpLayer, size_t tmpNumberOfNodes)
{
	bool tmpReturn = false;
	size_t tmpLayer = this->validLayer(iTmpLayer);
	// tmpLayer is in valid range
	this->vdNetworkTopology.at(max(0, tmpLayer - 1)) = max(1, tmpNumberOfNodes);
	return true;
}

bool AINetClass::resetCounter()
{
	this->iCounter = 0;
	return (this->iCounter == 0);
}

void AINetClass::TrainingDataColumnPush_Back(std::string tmpString)
{
	vTrainingDataColumns.push_back(tmpString);
}

std::vector<std::vector<double>> *AINetClass::getTrainingData()
{
	/** This is a wrapper function and returns a pointer to the training data matrix. 
		\return a pointer to the training data matrix
	*/
	return this->ptrAINDataContainer->ptrTrainingDataMatix();
}

void AINetClass::shuffleTrainingData()
{
	/** This function is used to shuffle the vector containing numeric references to rows of the training data set.
	*/
	std::random_shuffle(this->inputDataPullList.begin(), this->inputDataPullList.end());
}

std::string AINetClass::getTrainingDataOptionTimeString()
{
	/** Function returns a string "none", "date", "time" or "datetime" depending on what mode was set in training data.
		\return std::string 
	*/
	std::string strReturn = "none";
	switch (this->ptrAINDataContainer->getTimeMode())
	{
	case 2:
		strReturn = "date";
		break;
	case 3:
		strReturn = "time";
		break;
	case 4:
		strReturn = "datetime";
		break;
	default:
		strReturn = "none";
		break;
	}
	return strReturn;
}

void AINetClass::activateNetwork()
{
	/** Perform calculation using current values in the network.
	*/

	size_t numHiddenLayers = this->getNumberOfLayers(true);

	for (size_t currentLayer = 2; currentLayer <= 1 + this->getNumberOfLayers(true); currentLayer++)
	{
		// do this for each internal layer
		for (size_t h = this->getLayerStart(currentLayer); h <= this->getLayerStart(currentLayer, false); ++h)
		{
			// do this for each node in layer
			double weightedInput = 0.0;
			for (size_t p = this->getLayerStart(currentLayer - 1); p <= this->getLayerStart(currentLayer - 1, false); ++p)
			{
				// do this for each node in previous layer
				weightedInput += this->vecWeights[p][h] * this->vecValues[p];
			}
			// handle the thresholds
			weightedInput += (-1 * this->vecThresholds[h]);
			this->vecValues[h] = this->NodeFunction(weightedInput, h);
		}
	}

	for (size_t o = this->getLayerStart(-1); o <= this->getLayerStart(-1, false); ++o)
	{
		double weightedInput = 0.0;
		for (size_t d = this->getLayerStart(-2); d <= this->getLayerStart(-2, false); ++d)
		{
			weightedInput += this->vecWeights[d][o] * this->vecValues[d];
		}
		// handle the thresholds
		weightedInput += (-1 * this->vecThresholds[o]);
		this->vecValues[o] = this->NodeFunction(weightedInput, o);
	}
}

void AINetClass::setOptionCSV(bool bSetGerman)
{
	// set option csv german
	this->bOptionCSVGER = bSetGerman;
}

void AINetClass::setOptionDisplayAllNodes(bool bDisplayAll)
{
	// setting option display all nodes
	this->bOptionDisplayAllNodes = bDisplayAll;
}

void AINetClass::setOptionShuffle(bool bSetShuffle)
{
	// setOptionShuffle
	bOptionShuffle = bSetShuffle;
}

void AINetClass::setOptionSilent(bool bSilent)
{
	// setting option Silent e.g. for threaded operation
	this->bSilent = bSilent;
	this->bOptionDisplayAllNodes = !bSilent;
	this->bOptionIO = !bSilent;
}

void AINetClass::setOptionNoDeep(bool bSetNoDeep)
{
	// set option to prevent usage of deep network
	this->optionNoDeep = bSetNoDeep;
}

void AINetClass::setOptionIO(bool bSetIO)
{
	// changes status of option IO
	this->bOptionIO = bSetIO;
}

void AINetClass::setOptionWeight(bool bSetWeight)
{
	// setting option weight
	this->optionWeight = bSetWeight;
}

void AINetClass::setOptionStatus(bool bSetStatus)
{
	// settion option status
	this->bOptionStatus = bSetStatus;
}

void AINetClass::setOptionNodeFunction(size_t tmpNodeFunction)
{
	this->setActivationFunction(tmpNodeFunction);
}

void AINetClass::setOptionThreadCombinatingMode(size_t iTCMode)
{
	// set the option for splitting or combining threads
	this->iThreadedCombinationMode = min(10, max(0, iTCMode));
}

void AINetClass::setPercentVerification(double tmpPercentVerifiy)
{
	// setting the amount of verification data to be used
	if (tmpPercentVerifiy > 1)
	{
		tmpPercentVerifiy = tmpPercentVerifiy / 100.0;
	}
	this->dPercentVerification = min(0, max(0.9, tmpPercentVerifiy));
}

void AINetClass::setTrainingRow(size_t iTmpRow)
{
	// set the next training row 
	this->iCounter = min(this->getTrainingDataRowsMax(),max(0,iTmpRow));
	this->loadTrainingLine();
}

void AINetClass::calculateLine(size_t iTmpRow)
{
	/** calculate one row from training data set*/
	this->setTrainingRow(iTmpRow);
	this->loadTrainingLine();
	this->activateNetwork();
	this->printIO(-1.0);
}

void AINetClass::sortNetwork()
{
	/** This function is meant to sort the network, so multiple networks can be combined */

	// begin sorting at the end of the network (of course we won't sort output)
	for (size_t iSort = this->getNumberOfLayers(); iSort > 1; --iSort)
	{
		// reload network variables to tmp variables each layer.
		std::vector<double> vdTmpValues = this->vecValues;
		std::vector<double> vdTmpExpectedValues = this->vecExpectedValues;
		std::vector<std::vector<double>> vdTmpWeights = this->vecWeights;
		std::vector<double> vdTmpThresholds = this->vecThresholds;
		std::vector<double> vdTmpCalcDelta = this->vecCalcDelta;
		// Sum all the weights
		// to do that create a vector with sum
		std::vector<double> vdSumWeights(this->getNumberOfNodesInLayer(iSort - 1),0.0);
		std::vector<double> vdSumWeightsSorted(this->getNumberOfNodesInLayer(iSort - 1), 0.0);
		std::vector<size_t> viSortList(this->getNumberOfNodesInLayer(iSort - 1), 0);
		for (size_t iWeightLayer = this->getLayerStart(iSort - 1); iWeightLayer <= this->getLayerStart(iSort - 1, false); iWeightLayer++)
		{
			size_t i = iWeightLayer - this->getLayerStart(iSort - 1);
			// now get all the weights of the nodes in the previous layer
			for (size_t iNodeLayer = this->getLayerStart(iSort); iNodeLayer <= this->getLayerStart(iSort, false); iNodeLayer++)
			{
				// sum the weights (absolut)
				vdSumWeights.at(i) = vdSumWeights.at(i) + abs(this->vecWeights[iWeightLayer][iNodeLayer]);
			}
		}
		vdSumWeightsSorted = vdSumWeights;
		//sort
		std::sort(vdSumWeightsSorted.begin(), vdSumWeightsSorted.end());
		// now population sortlist
		
		for (size_t i = 0; i < vdSumWeightsSorted.size(); i++)
		{
			// start at minimum of vdSUmWeightsSorted
			for (size_t j = 0; j < vdSumWeights.size(); j++)
			{
				if (vdSumWeights.at(j) == vdSumWeightsSorted.at(i))
				{
					// found corresponding element
					viSortList.at(i) = j;
					// clearing sumweights at this point to prevent doubles being sorted incorrectly. calculation should never be -1 because it is caclulated as abs.
					vdSumWeights.at(j) = -1.0;
					break;
				}
			}
		}

		// prevent sorting of input layer
		if(iSort>2)
		{
			// crawl all the elements 
			size_t iBegin = this->getLayerStart(iSort - 1);
			size_t iNode = 0;
			for (size_t i = 0; i < this->getNumberOfNodesInLayer(iSort-1); i++)
			{
				iNode = i + iBegin;
				this->vecValues.at(iNode) = vdTmpValues.at(iBegin+viSortList.at(i));
				this->vecThresholds.at(iNode) = vdTmpThresholds.at(iBegin + viSortList.at(i));
				this->vecCalcDelta.at(iNode) = vdTmpCalcDelta.at(iBegin + viSortList.at(i));
				this->vecExpectedValues.at(iNode) = vdTmpExpectedValues.at(iBegin + viSortList.at(i));
				size_t jNode = 0;
				size_t jBegin = this->getLayerStart(iSort - 2);
				// sort weights to previous layer
				for (size_t j = 0; j < this->getNumberOfNodesInLayer(iSort - 2); j++)
				{
					jNode = jBegin + j;
					this->vecWeights.at(jNode).at(iNode) = vdTmpWeights[jNode][iBegin + viSortList.at(i)];
					//todo continue, but what?
				}
				// sort weights from previous layer
				jBegin = this->getLayerStart(iSort);
				for (size_t j = 0; j < this->getNumberOfNodesInLayer(iSort); j++)
				{
					jNode = jBegin + j;
					this->vecWeights.at(iNode).at(jNode) = vdTmpWeights[iBegin + viSortList.at(i)][jNode];
					//todo continue, but what?
				}
			}
		}
		// sorting is now finished
	}
}

void AINetClass::initialize(std::vector<size_t> iInternalTopology)
{
	/** This initializes the network. Topology is been loaded from training data.
		\param iInternalTopology (optional) This parameter can be used for setting internal network topolgy for the inner layers.
	*/

	// first of all the network topology has to be clear.
	if (iInternalTopology.size() == 1 && iInternalTopology.at(0) == 0)
	{
		// use the network provided by training data
		this->vdNetworkTopology = this->ptrAINDataContainer->getNetworkTopology();
	}
	else
	{
		this->vdNetworkTopology.clear();
		this->vdNetworkTopology.push_back(this->ptrAINDataContainer->getNetworkTopology().at(0));
		for (size_t intTemp = 0; intTemp < iInternalTopology.size(); ++intTemp)
		{
			this->vdNetworkTopology.push_back(iInternalTopology.at(intTemp));
		}
		this->vdNetworkTopology.push_back(this->ptrAINDataContainer->getNetworkTopology().at(this->ptrAINDataContainer->getNetworkTopology().size() - 1));
	}
	size_t tmpTotalNumberNodes = this->NUMNODES();
	
	this->iNumInputNodes= this->getNumberOfNodesInLayer(1);
	this->iNumRealInputNodes = this->getNumberOfNodesInLayer(1);
	this->iNumOutputNodes=this->getNumberOfNodesInLayer(-1);
	this->recalculateInputDataPullList();

	this->vecValues.clear();
	this->vecValues.reserve(tmpTotalNumberNodes);
	this->vecCalcDelta.clear();
	this->vecCalcDelta.reserve(tmpTotalNumberNodes);
	this->vecWeights.clear();
	this->vecWeights.reserve(tmpTotalNumberNodes);
	this->vecThresholds.clear();
	this->vecThresholds.reserve(tmpTotalNumberNodes);
	this->vecExpectedValues.clear();
	this->vecExpectedValues.reserve(tmpTotalNumberNodes);

	std::vector<double> tmpVector = { 0.0 };
	tmpVector.resize(tmpTotalNumberNodes, { 0.0 });
	this->vvErrors.clear();
	this->vvErrors.resize(this->ptrAINDataContainer->getTrainingDataRowsMax(), tmpVector);
	

	for (size_t i = 0; i <= tmpTotalNumberNodes; i++)
	{
		// even if vector <int> [0] is defined. it is not to be used.
		this->vecValues.push_back(0.0);
		this->vecCalcDelta.push_back(0.0);
		this->vecThresholds.push_back(0.0);
		this->vecExpectedValues.push_back(0.0);
		std::vector<double> tmpRow;
		for (size_t y = 0; y <= tmpTotalNumberNodes; y++) {

			tmpRow.push_back(0.0);
		}
		this->vecWeights.push_back(tmpRow);
	}


	this->initializationDone = true;
}

void AINetClass::saveResultingNetwork(size_t iNumber)
{
	// saving all data to a file
	size_t iMaxNodes = 0;
	std::string cResultingNetworkFileName;
	cResultingNetworkFileName = "-" + std::to_string(iNumber) + "-results-ainetwork.csv";
	iMaxNodes = (size_t)this->vecWeights.size();
	// output Weight to file
	std::ofstream fileResultingNetwork;
	time_t t = time(NULL);
	struct tm ts;
	char clocalerror[255] = "none";
	char clocalTime[80] = "";
	char cDefaultName[23] = "_results.ainetwork.csv";
	std::string tmpCurrentFormula = "";
	std::string tmpCurrentValue = "";
	
	fileResultingNetwork.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		localtime_s(&ts, &t);
		strftime(clocalTime, 80, "%Y-%m-%d_%H.%M.%S", &ts);
		cResultingNetworkFileName = clocalTime + cResultingNetworkFileName;
		
		/* DEBUG */
		OutputDebugStringA("saving data of thread:");
		OutputDebugStringA(this->strInternalName.c_str());
		OutputDebugStringA(" to file: ");
		OutputDebugStringA(cResultingNetworkFileName.c_str());
		OutputDebugStringA("\n");
		/* END DEBUG */

		fileResultingNetwork.open(cResultingNetworkFileName.c_str(), std::fstream::out | std::fstream::app);
		
		fileResultingNetwork << "output network " << this->strInternalName << " with formula\n";

		/* DEBUG */
		OutputDebugStringA("maximum number of layers:");
		OutputDebugStringA(std::to_string(this->getNumberOfLayers()).c_str());
		/* END DEBUG */

		std::string strFileContents = "";
		for (size_t iNode = 1; iNode <= this->getMaximumNodesLayer(true); iNode++)
		{
			// begin network and calculation output
			strFileContents = "";

			for (size_t iLayer = 0; iLayer <= this->getNumberOfLayers(); iLayer++)
			{
				size_t iNumNodesLayer = 0;
				iNumNodesLayer = this->getNumberOfNodesInLayer(max(1, iLayer));
				if (iNode > iNumNodesLayer)
				{
					break;
				}
				else
				{
					if (iLayer == 0)
					{
						// write name column
						strFileContents = this->ptrAINDataContainer->TrainingDataColumnName(iNode) + ",";
					}
					else if (iLayer == 1)
					{
						strFileContents = strFileContents + std::to_string(this->vecValues[iNode]) + ",";
					}
					else
					{
						size_t iNumLayerStart = this->getLayerStart(iLayer);
						if (iNode <= iNumLayerStart + this->getNumberOfNodesInLayer(iLayer))
						{
							// clear value an set it to theshold
							tmpCurrentValue = std::to_string(-1 * this->vecThresholds[iNumLayerStart - 1 + iNode]);
							//create formula for all nodes in previous layer
							size_t iNumNodesPrevLayer = this->getNumberOfNodesInLayer(iLayer - 1);
							size_t iNumPrevLayerStart = this->getLayerStart(iLayer - 1);
							for (size_t x = 1; x <= iNumNodesPrevLayer; x++)
							{
								tmpCurrentValue += "+" + this->getExcelColumn(iLayer) + std::to_string(x+1);// times 1 is due to first line content
								tmpCurrentValue += "*" + std::to_string(this->vecWeights[iNumPrevLayerStart -1+x][iNumLayerStart - 1 + iNode]);// times weight
							}

							strFileContents += this->NodeFunctionXLS((size_t)iNode,tmpCurrentValue);
							strFileContents += ",";		// end of cell
						}
					}
				}
			}
			this->generateFileOutput(strFileContents);
			fileResultingNetwork << strFileContents << "\n"; 	// write line to file
		}

		fileResultingNetwork << "--- Weight as result from " << cResultingNetworkFileName << " on " << clocalTime << "---\n";
		
		std::string tmpFileContents = "node,thresholds,";
		for (size_t i = 1; i < iMaxNodes; i++)
		{
			tmpFileContents = tmpFileContents + std::to_string(i) + ","; //enumerating x-axis
		}
		tmpFileContents = tmpFileContents + "\n"; // end of first line
		for (size_t y = 1; y < this->vecWeights.size(); y++)
		{
			tmpFileContents = tmpFileContents + "node" + std::to_string(y) + " to x," + std::to_string(this->vecThresholds[y]) + ","; //output first column to remind users of position of matrix
			for (size_t x = 1; x < this->vecWeights[y].size(); x++)
			{
				tmpFileContents = tmpFileContents + std::to_string(this->vecWeights[y][x]) + ",";//output the real data
			}
			tmpFileContents = tmpFileContents + "\n";// end of row;
		}
		tmpFileContents = tmpFileContents + "--- Weights end ---\n";
		this->generateFileOutput(tmpFileContents);
		fileResultingNetwork << tmpFileContents;

		// write the training data to file
		tmpFileContents = "-- training data begin -- \nRow,";
		// writing the header
		for (size_t iColumn = 1; iColumn < this->ptrAINDataContainer->getTrainingDataColumnsMax(); ++iColumn)
		{
			tmpFileContents = tmpFileContents + this->ptrAINDataContainer->TrainingDataColumnName(iColumn) + ",";
		}
		for (size_t iColumn = this->getLayerStart(-1); iColumn <= this->getLayerStart(-1,false); ++iColumn)
		{
			tmpFileContents = tmpFileContents + "Calculated_" +std::to_string(iColumn + (size_t)1) + ",";;
		}
		for (size_t iColumn = 0; iColumn <vvErrors.at(0).size(); iColumn++)
		{
			tmpFileContents = tmpFileContents + "Error_" + std::to_string(iColumn + (size_t) 1) + ",";
		}
		tmpFileContents = tmpFileContents + "\n";
		// writing the values
		for (size_t iTrainingLine = 0; iTrainingLine < this->ptrAINDataContainer->getTrainingDataRowsMax(); ++iTrainingLine)
		{
			tmpFileContents = tmpFileContents + std::to_string(iTrainingLine)+",";
			// write training data
			for(size_t iColumn=1; iColumn < this->ptrAINDataContainer->getTrainingRowSizeT(iColumn); iColumn++)
			{
				tmpFileContents = tmpFileContents + std::to_string(this->ptrAINDataContainer->getTrainingDataValue(iColumn,iTrainingLine))+ ",";
			}
			for (size_t iNode = this->getLayerStart(-1); iNode <= this->getLayerStart(-1, false); iNode++)
			{
				tmpFileContents = tmpFileContents + std::to_string(this->vecValues.at(iNode)) + ",";
			}

			// write errors, but check first if there is a error row left
			if (iTrainingLine < vvErrors.size())
			{
				for (size_t iRow = 0; iRow < vvErrors.at(iTrainingLine).size(); iRow++)
				{
					tmpFileContents = tmpFileContents + std::to_string(vvErrors.at(iTrainingLine).at(iRow)) + ",";
				}
			}
			tmpFileContents = tmpFileContents + "\n";
		}
		tmpFileContents = tmpFileContents + "-- training data end -- \n";
		this->generateFileOutput(tmpFileContents);
		fileResultingNetwork << tmpFileContents;

		for (size_t i = 0; i < this->errorList.size(); ++i)
		{
			fileResultingNetwork << this->errorList.at(i) << "\n";
		}

		fileResultingNetwork.close();
		this->vstrResultFilenames.push_back(cResultingNetworkFileName);
	}
	catch (...) {
		fileResultingNetwork.close();
		//TODO This is in conflict with Windows UWP App. 
		//this->throwFailure("Error while saving data." + strerror_s(clocalerror, errno), false);
	}
}

void AINetClass::setActivationFunction(size_t typeOfActivationFunction, size_t specificLayer)
{
	/** Is used to set the activation function of all layers or for a specific layer.
		\param typeOfActivationFunction selects the type of the activation function
		\param specificLayer (optional) to be used if this only applies to a specific layer
	*/
	if (specificLayer > 0)
	{
		// set activation function for specified layer
		this->viLayerActivationFunction.at(min(specificLayer-1, viLayerActivationFunction.size() - 1)) = typeOfActivationFunction;
	}
	else
	{
		this->iActivationFunction = typeOfActivationFunction;
		for (size_t i = 0; i < this->viLayerActivationFunction.size(); i++)
		{
			this->viLayerActivationFunction.at(i) = typeOfActivationFunction;
		}
	}
}

void AINetClass::setDataFileName(std::string strFileName)
{
	// set the file name for data input
	this->strAIDataFileName = strFileName;
}

void AINetClass::setInternalName(std::string strIntName)
{
	// set the internal name of the class
	this->strInternalName = strIntName;
}

void AINetClass::setOptionAutoGenerate(bool bAutoGenerate)
{
	// set option for automatic generation of internal network
	this->bOptionAutoGenerate = bAutoGenerate;
}

void AINetClass::connectNodes(bool bFullyConnected, size_t iRandSeed, bool bDeleteExisting)
{
	/** This function is used to create the initial connection of nodes.
		\param bFullyConnected is used to link al nodes from one layer with all nodes from the previous layer.
		\param iRandSeed is the seed for the random number generator.
		\param bDeleteExisting (optional) delete existing connections and reset all values.
	*/

	if (!this->initializationDone)
	{
		this->throwFailure("network not properly initialized", true);
	}

	if (!this->bHasBeenConnected || bDeleteExisting)
	{
		this->bHasBeenConnected = true;
		// first do the auto-generation if parameter is set
		this->autoGenerateInternalNetwork();

		// TODO allow smart connected network by removing next line of code
		bFullyConnected = true;
		//variables
		size_t tmpTotalNumberNodes = 0;

		srand((unsigned int)iRandSeed); // problems with conversion don't matter because it's random

		//function
		tmpTotalNumberNodes = this->NUMNODES();

		for (size_t x = 1; x <= tmpTotalNumberNodes; ++x) {
			if (bFullyConnected)
			{
				for (size_t y = 1; y <= tmpTotalNumberNodes; ++y) {
					// all connections are created, except connections to self
					if (x == y)
					{
						this->vecWeights[x][y] = 0.0;
					}
					else
					{
						// TODO: BUG RANDOM NUMBER is the same for every run of this funtion. get more randomness.
						this->vecWeights[x][y] = (rand() % 200) / 100.0;
					}
				}
			}
			else
			{
				// only valid & used connections are created
				for (size_t y = this->NUMINPUTNODES() + 1; tmpTotalNumberNodes; y++)
				{
					// generate node connections for all valid weights
					// next line is correct, but if wont work becuse vdNetworkTopology does not represent the network w/ historic data
					if (this->getLayerByNode(x) == this->getLayerByNode(y) - 1)
					{
						// node y is element of layer after node x
						this->vecWeights[x][y] = (rand() % 200 / 100.0);
					}
				}
			}
		}
		// generating thresholds for all nodes except input nodes
		for (size_t i = this->NUMINPUTNODES() + 1; i <= tmpTotalNumberNodes; i++)
		{
			this->vecThresholds[i] = rand() / (double)rand();
		}
	}
}

void AINetClass::combineNetworks(AINetClass *ptrAiNetClass, std::mutex & ptrMutex, size_t iNumber)
{
	// combining networks
	std::lock_guard<std::mutex> guard(ptrMutex);
	this->calculateErrorMSE(-1);
	ptrAiNetClass->calculateErrorMSE(-1);
	
	//TODO:BUG Check if sorting works
	ptrAiNetClass->sortNetwork();

	// TODO:DEBUG saving is just for debug
	ptrAiNetClass->saveResultingNetwork(iNumber);
	
	// combine date according to combination mode
	if (this->iThreadedCombinationMode == 0)
	{
		// combine all data
		// todo write data combination code.
	}

	Sleep(1000);
}

void AINetClass::createNetwork(std::vector<std::string> tmpsNetwork)
{
	// create network based on std::vector<std::string> first empty or zero line will end creation
	std::vector<size_t> tmpviNetwork;
	tmpviNetwork.resize(tmpsNetwork.size(), 0);
	for (size_t i = 0; i < tmpviNetwork.size(); i++)
	{
		tmpviNetwork.at(i) = atoi(tmpsNetwork.at(i).c_str());
	}
	this->createNetwork(tmpviNetwork);
}

void AINetClass::createNetwork(std::vector<size_t> tmpviNetwork)
{
	// create network based on std::vector<size_t> first line with zero will end creation
	size_t iNumLayers = 2;
	for (size_t i = 0; i < tmpviNetwork.size(); i++)
	{
		if (tmpviNetwork.at(i) > 0)
		{
			// set new maximum
			iNumLayers = max(iNumLayers, i);
			// continue
		}
		else
		{
			// this is zero so there cannot be any network.
			break;
		}
	}
	this->setNumberOfHiddenLayers(max(1, iNumLayers - 1));
	this->setNumInputNodes(tmpviNetwork.at(0));
	this->setNumOutputNodes(tmpviNetwork.at(iNumLayers));
	for (size_t i = 1; i < iNumLayers; i++)
	{
		// set the number of nodes in each hidden layer
		this->setNumberOfNodesinLayer(i+1, tmpviNetwork.at(i));
	}
}

void AINetClass::loadTrainingLine()
{
	/** Used to fetch training data from storage and put it into calculation space.
	First Copy input node data from matrix to input notes.
	Second perform training with data.
	*/
	size_t tmpCurrentRow = this->CurrentTrainingDataRow();

	for (size_t i = 0; i <= this->iNumRealInputNodes; i++)
	{
		this->vecValues[i] = this->ptrAINDataContainer->getTrainingDataValue(i,tmpCurrentRow);
	}

	// now adding historic data for the input
	for (size_t h = 1; h <= this->iTimePreviousRows; h++)
	{
		if (this->iTimeNumInputColumns == 0)
		{
			// for each previous row add all input node values
			for (size_t i = 1; i <= this->iNumRealInputNodes; ++i)  // starting from 1 to add data after input nodes.
			{
				this->vecValues[(h*this->iNumRealInputNodes+i)] = this->getTrainingDataValue(tmpCurrentRow + h, i);
			}
		}
		else
		{
			// for each row select the first (iTimeNumInputColumns) columns.
			for (size_t i = 1; i <= this->iTimeNumInputColumns; ++i) // starting from 1 to add data after input nodes.
			{
				this->vecValues[this->iNumRealInputNodes+((h-1)*this->iTimeNumInputColumns) + i] = this->getTrainingDataValue(tmpCurrentRow + h, i);
			}
		}
		// TODO add handling for previous/historic data
	}

	// setting output nodes
	for (size_t i = 0; i < this->getNumberOfOutputNodes(); ++i)
	{
		this->vecExpectedValues[this->getLayerStart(-1) + i] = this->getTrainingDataValue(tmpCurrentRow,this->iNumRealInputNodes+1+i);
	}
}

void AINetClass::trainNetwork(bool bSilent)
{
	/** This function is used to train the network. Must not be called bevore AINetClass::initialize()
		\param bSilent (optional) This parameter can be used to prevent any output to screen.
	*/

	if (bSilent)
	{
		this->bOptionDisplayAllNodes = false;
		this->bOptionStatus = false;
	}
	while (this->continueCalculation())
	{
		this->loadTrainingLine(); 
		this->activateNetwork();

		double sumOfSquaredErrors = 0.0;

		sumOfSquaredErrors = this->updateWeights();

		// calculate the Worst error and output
		if (this->IsTrainingEndOfDataset())
		{
			// printf("Max Error at Row %zu with value %8.6f\n", iWorstErrorRow, dWorstError);
			iWorstErrorRow = 0;
			dWorstError = 0.0;
		}
		else if (max(dWorstError, sumOfSquaredErrors) == sumOfSquaredErrors)
		{
			iWorstErrorRow = this->Counter() % this->getTrainingDataRowsMax();
			dWorstError = max(dWorstError, sumOfSquaredErrors);
		}
		this->printIO(sumOfSquaredErrors); // if options set displaying all net

		if (this->bOptionDisplayAllNodes)
		{
			this->displayAllNodes(sumOfSquaredErrors); // if options ist set displaying all nodes
		}
		this->Counter(true);
	}
}

void AINetClass::printIO(double sumOfSquaredErrors)
{
	COORD ord;
	if (this->bOptionIO)
	{
		if (this->IsTrainingEndOfDataset())
		{
			printf("\nNew Row    |");
			for (size_t i = 1; i <= this->NUMREALINPUTNODES() + this->NUMOUTPUTNODES(); i++)
			{
				if (this->TrainingDataColumns() >= i)
				{
					printf("%s|", this->ptrAINDataContainer->TrainingDataColumnName(i).c_str());
				}
			}
			printf("\n");
		}
		printf("%8zu:Row: %8zu|", this->Counter(false), this->CurrentTrainingDataRow());
		for (size_t i = this->getLayerStart(1); i <= this->getLayerStart(1,false); i++)
		{
			//listing all input nodes
			if (i <= this->NUMREALINPUTNODES())
			{
				printf("%4.4f|", this->vecValues[i]);
			}
			else
			{
				printf("%4.4f!", this->vecValues[i]);
			}
		}
		for (size_t i = this->getLayerStart(-1); i <= this->getLayerStart(-1,false); i++)
		{
			//listing all outputnodes nodes
			printf("%8.4f", this->vecValues[i]);
			printf("(%8.4f)|", this->vecExpectedValues[i]);
		}
		printf("err:%8.5f", sumOfSquaredErrors);
		// if historic data is used create a new line for each of them
		printf("\n");
	}
	else
	{
		ord.X = 0;
		ord.Y = 0;
		if (this->IsTrainingEndOfDataset())
		{
			printf("\n");
		}
		if (sumOfSquaredErrors > 1)
		{
			printf("|");
		}
		else if (sumOfSquaredErrors > 0.1)
		{
			printf("l");
		}
		else if (sumOfSquaredErrors > 0.01)
		{
			printf("i");
		}
		else if(sumOfSquaredErrors > 0.001)
		{
			printf(",");
		}
		else
		{
			printf(".");
		}
		SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), ord);
	}
}

void AINetClass::displayWeights()
{
	size_t iMaxNodes = 0;
	iMaxNodes = (size_t) this->vecWeights.size();
	// TODO: optionsWeight not mplemented yet
	if (this->optionWeight)
	{
		printf("--- Weights ---\n");
		// output Weight to screen
		printf("node;thresholds;");
		for (size_t i = 1; i <= iMaxNodes; i++)
		{
			printf("%zu;", i);//enumerating x-axis
		}
		printf("\n"); // end of first line
		for (size_t y = 1; y < this->vecWeights.size(); y++)
		{
			printf("from node %zu to x;", y); //output first column to remind users of position of matrix
			printf("%8.4f;", this->vecThresholds[y]);
			for (size_t x = 1; x < this->vecWeights[y].size(); x++)
			{
				printf("%8.4f;", this->vecWeights[y][x]); //output the real data
			}
			printf("\n");// end of row;
		}
		printf("--- Weights end ---\n");
	}
}

void AINetClass::displayStatus()
{
	// display status message
	if (this->bOptionStatus)
	{
		printf("\n--- STATUS ---\nNUMINPUTNODES=\t%zu(%zu)\n", this->NUMREALINPUTNODES(), this->NUMINPUTNODES());
		printf("NUMOUTPUTNODES=\t%zu\n", this->NUMOUTPUTNODES());
		printf("NUMNODES=\t%zu\n", this->NUMNODES());
		printf("MAXITERATIONS=\t%zu, of which %zu have been performed\n", this->getMaxIterations(), this->Counter()); //131072;
		printf("LEARNINGRATE=\t%8.4f\n", this->LearningRate());
		printf("training lines = \t%8zu\n", this->getTrainingDataRowsMax());
		printf("-- Options --\n");
		printf("SHUFFLE:\t%s\n", this->bOptionShuffle ? "true" : "false");
		printf("AUTO-GENERATE:\t%s\n",this->bOptionAutoGenerate ? "true" : "false");
		printf("CSV-GER:\t%s\n", this->bOptionCSVGER ? "true" : "false");
		printf("passes=\t%8zu\t with %8zu additional rows\n", this->getMaxIterations() / this->getTrainingDataRowsMax(), this->getMaxIterations() % this->getTrainingDataRowsMax());
		if (this->iTimeNextRows == 0 && this->iTimePreviousRows == 0)
		{
			printf("TIME_DEPENCY:\tOFF\n");
		}
		else
		{
			printf("TIME_DEPENCY:\tON\n\tin\t%4zu rows with first %4zu nodes\n\tout\t%4zu rows with first %4zu nodes", this->iTimePreviousRows, this->iTimeNumInputColumns, this->iTimeNextRows, this->iTimeNumOutputColumns);
		}
		printf("\nneural network with %8zu layers", this->getNumberOfLayers());
		for (size_t numLayers = 1; numLayers <= this->getNumberOfLayers(); numLayers++)
		{
			printf("\nlayer %4zu with %8zu nodes, from node %4zu to %4zu", numLayers, this->getNumberOfNodesInLayer(numLayers), this->getLayerStart(numLayers), this->getLayerStart(numLayers, false));
			if (numLayers == 1)printf(" (input) ");
			if (numLayers == this->getNumberOfLayers())printf(" (output) ");
		}
		printf("--- STATUS ---\n");
	}
}

void AINetClass::displayAllNodes(double sumOfSquaredErrors)
{
	// display all nodes
	if (this->IsTrainingEndOfDataset())
		printf("Display Whole Network and Error---------------------\n");
	
	// get maximum number of nodes (maximum rows)
	for (size_t i = 1; i <= this->getMaximumNodesLayer(true); i++)
	{
		// get maximum number of layers (max cols)
		for (size_t j = 1; j <= this->getNumberOfLayers();  j++)
		{
			// do the output magic
			if (i <= this->getNumberOfNodesInLayer(j))
			{
				printf("%2zu:%8.4f|", (this->getLayerStart(j) + i - 1), this->getNodeValue(this->getLayerStart(j) + i -1));
			}
			else
				printf("--:---.----|");
		}
		printf("\n");
	}
	printf("\n------ err: %8.5f\n", sumOfSquaredErrors);
}

std::string AINetClass::getDataFileName()
{
	if (this->strAIDataFileName == "")
		return "default.csv";
	else
		return this->strAIDataFileName;
}

std::string AINetClass::getAIDataFileHeader()
{
	/* returns name as given in data file, does not return the actual filename */
	return this->strAIDataFileHeader;
}

std::string AINetClass::getExcelColumn(size_t stColumn)
{
	// this converts a number of columns to excel alphabetic numbering
	std::string abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	std::string returnString = "";
	if (max(1,stColumn) <= 26)
	{
		// only on letter is returned
		returnString = abc.at(stColumn - 1);
	}
	else if (stColumn <= 676)
	{
		// 676 is 26*26 aka ZZ ;-)
		returnString = abc.at((int)(stColumn / 26)); // yes it cuts of the end
		returnString = returnString + abc.at(stColumn % 26);
	}
	return returnString;
}

double AINetClass::updateWeights()
{
	double sumOfSquaredErrors = 0.0;
	// OLD:
	// sumOfSquaredErrors = aincNetwork.updateWeights();
	// NEW:
	for (size_t iLayer = this->getNumberOfLayers(); iLayer >=1; --iLayer)
	{
		sumOfSquaredErrors += this->updateWeightsInLayer(iLayer);
	}
	return sumOfSquaredErrors;
}

double AINetClass::getVersion()
{
	/** this function returns the Version of the AINetClass
		\return double Version
	*/
	return this->dVersion;
}

double AINetClass::getNodeValue(size_t tmpNode)
{
	// returns value of node
	size_t iReturn;
	iReturn = (size_t) min(max(0, tmpNode),this->vecValues.size()-1);
	if (iReturn==tmpNode)
	{
		return this->vecValues.at(iReturn);
	}
	else
	{
		return 0.0;
	}
}

std::vector<std::string> AINetClass::getErrorList()
{
	// this one returns the error list
	return this->errorList;
}

std::vector<std::string> AINetClass::splitString(const std::string & strInput, const std::string & strDelimiter)
{
	// spliting String
	std::vector<std::string> strElements;

	for (size_t stStart = 0, stEnd; stStart < strInput.length(); stStart = stEnd + strDelimiter.length())
	{
		size_t stPosition = strInput.find(strDelimiter, stStart);
		stEnd = stPosition != std::string::npos ? stPosition : strInput.length();

		std::string strElement = strInput.substr(stStart, stEnd - stStart);

		strElements.push_back(strElement);

	}

	if (strInput.empty() || (strInput.substr(strInput.size()-strDelimiter.size(), strDelimiter.size()) == strDelimiter))
	{
		strElements.push_back("");
	}

	return strElements;
}

bool AINetClass::IsNetworkReady()
{
	//this one should test if network is ready to run.
	return this->initializationDone;
}

bool AINetClass::autoGenerateInternalNetwork()
{
	/** This function is generating the internal network topology based on proportional distribution of nodes from input to output.
		\return true if successfull.
	*/
	size_t iAutoInput = this->NUMINPUTNODES();
	size_t iAutoOutput = this->NUMOUTPUTNODES();
	size_t iAutoLayer = (size_t)this->vdNetworkTopology.capacity();
	double iAutoResult = 1;
	if (this->bOptionAutoGenerate)
	{
		// option auto generate is set
		for (size_t i = 2; i < iAutoLayer; i++)
		{
			iAutoResult = max(1,min(iAutoInput, round(iAutoInput - 0.8 * ((iAutoInput - iAutoOutput) / iAutoLayer * i))));
			this->setNumberOfNodesinLayer(i, (size_t)iAutoResult);
		}
		this->resizeVectors();
		return true;
	}
	else
	{
		// return false if network is not automaticaly generated
		return false;
	}
}

double AINetClass::getTrainingDataValue(size_t row, size_t column)
{
	/** This function is used to access the training data in a safe way 
		\param row select this row
		\param column select this column
		*/
	double tmpReturn=0;
	if (row < this->ptrAINDataContainer->getTrainingDataRowsMax())
	{
		if (column < this->ptrAINDataContainer->getTrainingRowSizeT(row))
		{
			tmpReturn= this->ptrAINDataContainer->getTrainingDataValue(column,row);
		}
		else
		{
			this->throwFailure("column exeeded", true);
		}
	}
	else
	{
		this->throwFailure("row exeeded", true);
	}
	return tmpReturn;
}

double AINetClass::updateWeightsInLayer(signed int iTmpLayer)
{
	/** Update the weights in the specified layer
		\param tmpLayer The layer in question.
	*/
	size_t tmpLayer = this->validLayer(iTmpLayer);
	size_t tmpLayerBegin = this->getLayerStart(tmpLayer);
	size_t tmpLayerEnd = this->getLayerStart(tmpLayer, false);
	
	double sumOfSquaredError = 0.0;
	size_t n = 0;

	for (size_t iCurrentNode = tmpLayerBegin; iCurrentNode <= tmpLayerEnd; ++iCurrentNode)
	{
		double dErrorAtCurrentNode = 0.0;
		// calculation of delta
		if (tmpLayer <= 1)
		{
			this->bBackpropagationActive = false;
		}
		else if(this->IsLastLayer(tmpLayer))
		{
			this->bBackpropagationActive = true;
			// calculate absolute error from output
			dErrorAtCurrentNode = this->vecValues[iCurrentNode] - this->vecExpectedValues[iCurrentNode];
			this->vvErrors.at(this->CurrentTrainingDataRow()).at(iCurrentNode - tmpLayerBegin) = dErrorAtCurrentNode;
			sumOfSquaredError += pow(dErrorAtCurrentNode, 2);
			// calculation of delta at current node
			double deltaAtNode = this->NodeFunction(this->vecValues[iCurrentNode], iCurrentNode, true) * dErrorAtCurrentNode;
			// saving delta at current node
			this->vecCalcDelta[iCurrentNode] = deltaAtNode;
		}
		else
		{
			this->bBackpropagationActive = true;
			// calculate error from following layer
			for (size_t iLaterNode = this->getLayerStart(tmpLayer + 1); iLaterNode <= this->getLayerStart(tmpLayer + 1, false); iLaterNode++)
			{
				// correcting the weights at first
				dErrorAtCurrentNode += this->vecCalcDelta[iLaterNode] * this->vecWeights[iCurrentNode][iLaterNode];
			}
			double deltaAtNode = this->NodeFunction(this->vecValues[iCurrentNode], iCurrentNode, true) * dErrorAtCurrentNode;
			this->vecCalcDelta[iCurrentNode] = deltaAtNode;
		}
		// calculaton of weights from all nodes in previous layer to iCurrentNode
		if (tmpLayer <= 1)
		{
			// there are no weights to input nodes.
		}
		else
		{
			for (size_t iPreviousNode = this->getLayerStart(tmpLayer - 1); iPreviousNode <= this->getLayerStart(tmpLayer - 1, false); iPreviousNode++)
			{
				double deltaWeight = -1 * this->LearningRate() * this->vecCalcDelta[iCurrentNode] * this->vecValues[iPreviousNode];
				this->vecWeights.at(iPreviousNode).at(iCurrentNode) += deltaWeight;
				// adjusting thresholds
				this->vecThresholds[iCurrentNode] += this->LearningRate() * this->vecCalcDelta[iCurrentNode];
			}
		}
		n = iCurrentNode;
	}
	return(sumOfSquaredError / max(n, 1));
}


size_t AINetClass::getNumberOfInputNodes()
{
	/** This function is a wrapper function to the training data container
		\return returns the number of input nodes.
	*/

	return this->ptrAINDataContainer->getNumberOfInputNodes();
}

size_t AINetClass::getNumberOfOutputNodes()
{
	/** This function is a wrapper function to the training data container
		\return returns the number of output nodes.
	*/
	return this->ptrAINDataContainer->getNumberOfOutputNodes();
}

std::string AINetClass::generateFileOutput(std::string& strFileContents)
{
	// convert data to suitable format
	if (this->bOptionCSVGER)
	{
		CodeFromWeb::ReplaceAllStrings(strFileContents, ",", ";");
		CodeFromWeb::ReplaceAllStrings(strFileContents, ".", ",");
	}
	return strFileContents;
}

std::string AINetClass::generateFileInput(std::string & strFileContents)
{
	// convert data to suitable format
	if (this->bOptionCSVGER)
	{
		CodeFromWeb::ReplaceAllStrings(strFileContents, ",", ".");
		CodeFromWeb::ReplaceAllStrings(strFileContents, ";", ",");
	}
	return strFileContents;
}

bool AINetClass::IsDoubleCritical(double dToBeClassified)
{
	/** This function is used to check if this is a critical floating point number
		\param dToBeClassified Number to be checked
		\return True for critical number
	*/
	switch (std::fpclassify(dToBeClassified)) {
		case FP_INFINITE: return true;
		case FP_NAN: return true;
		case FP_SUBNORMAL: return true;
		case FP_ZERO: return false;
		case FP_NORMAL: return false;
		default: return false;
	}
}

std::string AINetClass::IsDoubleCritical(double dToBeClassified, std::string sText)
{
	/** This function is used to check if this is a critical floating point number
		\param dToBeClassified Number to be checked
		\param sText text to be attatched
		\return True for critical number
	*/
	switch (std::fpclassify(dToBeClassified)) {
		case FP_INFINITE: return sText + "INFINITE";
		case FP_NAN: return sText + "NOT A NUMBER";
		case FP_SUBNORMAL: return sText + "SUBNORMAL";
		case FP_ZERO: return sText + "ZERO";
		case FP_NORMAL: return sText + "NORMAL";;
		default: return sText + "NORMAL";;
	}
	return std::string();
}

bool AINetClass::recalculateInputDataPullList()
{
	if (this->ptrAINDataContainer->getTrainingDataRowsMax() > 0)
	{
		this->inputDataPullList.clear();
		this->inputDataPullList.resize(this->ptrAINDataContainer->getTrainingDataRowsMax() - iTimePreviousRows);
		std::fill(this->inputDataPullList.begin(), this->inputDataPullList.end(), 0);
		for (size_t i = 0; i < this->inputDataPullList.capacity(); i++)
		{
			if (iTimePreviousRows < 0)
				this->inputDataPullList.at(i) = i - iTimePreviousRows; // add (2x negative) the offset at the beginning
			else
			{
				this->inputDataPullList.at(i) = i;	// simple generation of list
			}
		}
	}
	return (this->ptrAINDataContainer->getTrainingDataRowsMax() > 0);
}

bool AINetClass::throwFailure(std::string tmpError,bool doexit)
{
	// there is a failure in the network. 
	// i have to exit
	this->errorList.push_back(tmpError);
	if (doexit)
		exit(1);
	return true;
}

double AINetClass::calculateErrorMSE(int iTmpLayer)
{
	// this function calculates the mean square error for specified layer
	size_t iLayer = this->validLayer(iTmpLayer);
	double sumOfError = 0.0;
	double n = 0.0;
	for (size_t iCurrentNode = this->getLayerStart(iLayer); iCurrentNode <= this->getLayerStart(iLayer,false); iCurrentNode++)
	{
		sumOfError = pow(this->vecCalcDelta.at(iCurrentNode),2);
		n += 1;
	}
	if (n == 0.0)
		this->throwFailure("division by zero (n) in MSE Error calculation", true);
	return (sumOfError/n);
}

size_t AINetClass::resizeVectors()
{
	// recalculates the size of input vector()
	size_t tmpInputVectorSize = 0;
	if (this->iTimeNumInputColumns == 0)
	{
		tmpInputVectorSize += this->iNumRealInputNodes * (1 + this->iTimePreviousRows);
	}
	else
	{
		tmpInputVectorSize += this->iNumRealInputNodes + this->iTimeNumInputColumns * this->iTimePreviousRows;
	}
	this->vdNetworkTopology.at(0) = tmpInputVectorSize;
	for (size_t i = 1; i < vdNetworkTopology.size(); i++)
	{
		tmpInputVectorSize += vdNetworkTopology.at(i);
	}
	
	this->initialize();

	return tmpInputVectorSize;
}

size_t AINetClass::validLayer(signed int tmpLayer)
{
	// returning a valid layer between 1 and vdNetworkTopology.size()
	//Todo rewrite to make compatible with size_t 

	size_t retLayer = 0;
	if (tmpLayer < 0)
	{
		// first of all, make it positive.
		// if tmpLayer is negative, it is counted from the last layer in the network
		// due to the fact that it has to be -1 or smaler, this->vdNetworkTopology.size() is always substractet at least 1 element which is added so -1 refers to last layer.
		tmpLayer = max(1, (int) this->vdNetworkTopology.size() + 1 + tmpLayer);
	}
	else if (tmpLayer > 0)
	{
		// great it is greater than 0, so the first (aka input layer ist 1)
		tmpLayer = min((int) this->vdNetworkTopology.size(), tmpLayer);
	}
	else
	{
		// refering to first layer
		tmpLayer = 1;
	}

	retLayer = (size_t)tmpLayer; // force to size_t
	if (retLayer != (size_t)tmpLayer) // verify if conversion was successfull
	{
		retLayer = 1;
	}
	return retLayer;
}

size_t AINetClass::validLayer(size_t& tmpLayer)
{
	/** This function will return a valid layer.
		\param tmpLayer this is the layer to be tested.
		\return Returns a vaild layer. (1 to maximum number of layers)
	*/
	// returning a valid layer between 1 and vdNetworkTopology.size()
	if (tmpLayer > 0)
	{
		// great it is greater than 0, so the first (aka input layer ist 1)
		tmpLayer = min((int)this->vdNetworkTopology.size(), tmpLayer);
	}
	else
	{
		// refering to first layer
		tmpLayer = 1;
	}
	return tmpLayer;
}

double AINetClass::NodeFunction(double weightedInput, size_t currentNodeID, bool derivative)
{
	/** Calculates the output of a given node
		\param weightedInput Input of this node
		\param currentNodeID a specific node
		\param derivative (optional) set to true if calculation is performed backwards
	*/
	double dActivationFunctionResult = 0.0;

	switch (this->getActivationFunction(currentNodeID))
	{
	case 1:
		// using tanh
		if (derivative) { dActivationFunctionResult = 1 / pow(cosh(-weightedInput), 2); } 
		else { dActivationFunctionResult = tanh(weightedInput); }
		break;
	case 2:
		//BIP
		if (derivative) { dActivationFunctionResult = 2 * pow(E, weightedInput) / pow((pow(E, weightedInput) + 1), 2); }
		else { dActivationFunctionResult = (1 - pow(E, -weightedInput)) / (1 + pow(E, -weightedInput)); }
		break;
	case 3:
		// using rectified linear unit
		if (derivative) { dActivationFunctionResult = 1; }
		else { dActivationFunctionResult = max(0, weightedInput); }
		break;
	case 4:
		// 
		if (derivative) { dActivationFunctionResult = pow(E, -weightedInput) / pow((pow(E, -weightedInput) + 1), 2); }
		else { dActivationFunctionResult = 1.0 / (1.0 + pow(E, -weightedInput)); }
		break;
	default:
		if (derivative) { dActivationFunctionResult = pow(E, -weightedInput) / pow((pow(E, -weightedInput) + 1), 2); }
		else { dActivationFunctionResult = 1.0 / (1.0 + pow(E, -weightedInput)); }
		break;
	}
	return dActivationFunctionResult;
}

std::string AINetClass::NodeFunctionXLS(size_t tmpNode, std::string tmpCalculatedInput)
{
	// calculate xls output
	std::string myActivationFunction = "";
	size_t tmpActivationFunction = this->getActivationFunction(tmpNode);
	switch (tmpActivationFunction)
	{
	case 1:
		// using tanh
		myActivationFunction = "=TANHYP(%d)";
		break;
	case 2:
		//BIP
		myActivationFunction = "=(1-EXP(-1*(%d)))/(1+EXP(-1*(%d)))";
		break;
	case 3:
		//reLu
		myActivationFunction = "=MAX(0,%d)";
		break;
	case 4:
		// using linear activation function on output layer
		myActivationFunction = "=1.0/(1.0+EXP(-1*(%d)))";
		break;
	default:
		myActivationFunction = "=1.0/(1.0+EXP(-1*(%d)))";
		break;
	}

	CodeFromWeb::ReplaceAllStrings(myActivationFunction, "%d", tmpCalculatedInput);
	//ReplaceAllStrings(myActivationFunction, "--", "");

	return myActivationFunction;
}

std::string AINetClass::NodeFunctionJS(size_t tmpNode, std::string tmpCalculatedInput)
{
	// TODO WRITE CODE for NodeFunctionJS
	return std::string();
}
