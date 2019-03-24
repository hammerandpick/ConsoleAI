#include "pch.h"
#include "stdafx.h"
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "CodeFromWeb.h"
#include "AINetTrainingData.h"


AINetTrainingData::AINetTrainingData()
{
	/** Constructor for AINetTrainingData
	  * \relates AINetTrainingData(double tdm)
	*/
}

AINetTrainingData::~AINetTrainingData()
{
	this->vvTrainingDataMatrix.clear();
	this->vdNetworkTopology.clear();
}


size_t AINetTrainingData::getTrainingDataRowsMax(bool bReload)
{
	/** This function returns the maximum number of training data rows. It calculates the maximum number of rows from the file.
		For each line udes as previous or next data (historic calculation) this number will be reduced.
		Further reduction is done if verification data is inside the file.
		\return the maximum number of training data rows.
	*/

	if (this->intTrainingDataRowsMax == 0 || bReload)
	{
		this->intTrainingDataRowsMax = llround((this->vvTrainingDataMatrix.size() - this->intTimeNextRows - this->intTimePreviousRows)*(1 - this->dPercentVerificationData));
	}

	return this->intTrainingDataRowsMax;
}

size_t AINetTrainingData::getTrainingDataColumnsMax(bool bRecount)
{
	/** This function returns the number of columns in training data.
		\param bRecount (optional) if set true, this function will do a full count. if not set this function will rely on previously stored value.
		\return The number of columns in training data.
	*/

	size_t intMaxColumns = 0;

	if ((this->intTrainingDataColumsMax == 0) || bRecount)
	{
		// crawl all date for longest column
		for (size_t i = 0; i < this->vvTrainingDataMatrix.size(); ++i)
		{
			size_t intTempSize = this->vvTrainingDataMatrix.at(i).size();
			if (intTempSize > intMaxColumns)
			{
				intMaxColumns = intTempSize;
			}
		}
	}
	else
	{
		intMaxColumns = this->intTrainingDataColumsMax;
	}
	return intMaxColumns;
}

size_t AINetTrainingData::getTrainingDataBegin()
{
	/** This function is used to calculate the start of the training data. It will skip the number of defined previous rows, which will be used in calculation. Data has to be in ascending order.
	*/
	return (size_t)this->intTimePreviousRows;
}

size_t AINetTrainingData::getTrainingDataEnd()
{
	/** This function is used to calculate the end of the training data. It will skip the number of defined next rows, which will be used in calculation. Data has to be in ascending order.
	*/
	return (size_t)this->vdNetworkTopology.size() - (size_t)this->intTimeNextRows;
}

size_t AINetTrainingData::getTotalLines()
{
	return this->intLinesRead;
}

size_t AINetTrainingData::getNumberOfInputNodes()
{
	/** This returns the number of input nodes from the network topology
		\return Number of input nodes.	
	*/
	return this->vdNetworkTopology.front();
}

size_t AINetTrainingData::getNumberOfOutputNodes()
{
	/** This function returns the number of output nodes from network topology of training data file
		\return The number of output nodes.
	*/
	return this->vdNetworkTopology.back();
}

size_t AINetTrainingData::getTimeMode()
{
	/** Function returns AINetTrainingData->intTimeDataMode 
		\return 1 for none\n 2 for date \n 3 for time \n 4 for date and time */
	return this->intTimeDataMode;
}

double AINetTrainingData::getTrainingDataValue(size_t column, size_t row)
{
	/** This function returns the value of the training data in \p colum of \p row.
		\param column The colum to be returned.
		\param row The row to be retrned.
		\return Value of \p column and \p row.
	*/
	double dReturn = 0.0;
	if (row < this->vvTrainingDataMatrix.size())
	{
		if (column < this->vvTrainingDataMatrix.at(row).size())
		{
			dReturn= this->vvTrainingDataMatrix.at(row).at(column);
		}
		else
		{
			std::cerr << "column out of range: " << column;
		}
	}
	else
	{
		std::cerr << "row out of range: " << row;
	}
	return dReturn;
}

size_t AINetTrainingData::getTrainingRowSizeT(size_t row)
{
	/** This will return the size of the specified \p row.
		\param row The row of interest.
		\return The size of the specified \p row or zero if row is out of bounds.
	*/

	size_t intReturn = 0;
	if (row < this->vvTrainingDataMatrix.size())
	{
		intReturn = this->vvTrainingDataMatrix.at(row).size();
	}
	return intReturn;
}

std::vector<size_t> AINetTrainingData::getNetworkTopology()
{
	/** This function will return the topology from training data file.
		\return The topology of the network from training data file.
	*/
	return this->vdNetworkTopology;
}

std::vector<std::vector<double>> AINetTrainingData::getTrainingDataMatrix()
{
	/** This function returns the training data as new object. This should not be used. TODO: Check and delete.
		\return a new training data object.
	*/
	return this->vvTrainingDataMatrix;
}

std::vector<std::vector<double>>* AINetTrainingData::ptrTrainingDataMatix()
{
	/** This function returns the training data as new object. This should not be used. TODO: Check and delete.
		\return a new training data object.
	*/
	return &this->vvTrainingDataMatrix;
}

std::string AINetTrainingData::getTrainingDataFileName()
{
	/** This will return the file name of the training data file.
		\return String of file name.
	*/
	return this->strAIDataFileName;
}

std::string AINetTrainingData::setTrainingDataFileName(std::string strFileName)
{
	/** This will set the file name of the training data file. 
		\return String of file name.
	*/
	// TODO: add some verification here.
	this->strAIDataFileName;
	return this->strAIDataFileName;
}

bool AINetTrainingData::setOptionCSVGermanStyle(bool bGerStyle)
{
	/** This is used to set the option to convert a german style *.csv into a standard *.csv 
		\param bGerStyle if a german csv file is to be loaded this has to be set to true
		\return returns input parameter if set correctly.
	*/
	this->bOptionCSVGER = bGerStyle;
	return this->bOptionCSVGER;
}

bool AINetTrainingData::setPreferredNetworkTopology(std::string strPref)
{
	/** this function is used to set the prefered network topology
		\param strPref is a string with the topology as integer values from input (lowest) to output (highest)
		\return returns true if successfull
	*/

	std::string strDelimiter = ",";
	std::vector<std::string> strElements;

	for (size_t stStart = 0, stEnd; stStart < strPref.length(); stStart = stEnd + strDelimiter.length())
	{
		size_t stPosition = strPref.find(strDelimiter, stStart);
		stEnd = stPosition != std::string::npos ? stPosition : strPref.length();

		std::string strElement = strPref.substr(stStart, stEnd - stStart);

		strElements.push_back(strElement);

	}

	if (strPref.empty() || (strPref.substr(strPref.size() - strDelimiter.size(), strDelimiter.size()) == strDelimiter))
	{
		strElements.push_back("");
	}

	return false;
}

bool AINetTrainingData::setPreferredNetworkTopology(std::vector<size_t> vsPref)
{
	/** this function is used to set the prefered network topology
		\param vsPref is a vector<size_t> with the topology as integer values from input (lowest) to output (highest)
		\return returns true if successfull
	*/
	this->vdNetworkTopology = vsPref;
	return (this->vdNetworkTopology == vsPref);
}

std::vector<size_t> AINetTrainingData::splitStringToSizeT(const std::string & strInput, const std::string & strDelimiter)
{
	/** this function i used to split a string into a vector of integers.
		\param strInput is a string of elements separated by \p strDelimiter
		\param strDelimiter is string, which is used to split \p strInput into pieces.
		\return is returning the content of \p strInput converted into a vector of integers.
	*/
	std::vector<size_t> strElements;

	for (size_t stStart = 0, stEnd; stStart < strInput.length(); stStart = stEnd + strDelimiter.length())
	{
		size_t stPosition = strInput.find(strDelimiter, stStart);
		stEnd = stPosition != std::string::npos ? stPosition : strInput.length();

		std::string strElement = strInput.substr(stStart, stEnd - stStart);

		if (atoi(strElement.c_str())>=1)
		{
			// do not pushback if layer is empty (0).
			strElements.push_back(atoi(strElement.c_str()));
		}
	}

	if (strInput.empty() || (strInput.substr(strInput.size() - strDelimiter.size(), strDelimiter.size()) == strDelimiter))
	{
		// do nothing
	}

	return strElements;
}

std::vector<double> AINetTrainingData::splitStringToDouble(const std::string & strInput, const std::string & strDelimiter)
{
	/** this function i used to split a string into a vector of doubles.
		\param strInput is a string of elements separated by \p strDelimiter
		\param strDelimiter is string, which is used to split \p strInput into pieces.
		\return is returning the content of \p strInput converted into a vector of doubles.
	*/
	std::vector<double> strElements;

	for (size_t stStart = 0, stEnd; stStart < strInput.length(); stStart = stEnd + strDelimiter.length())
	{
		size_t stPosition = strInput.find(strDelimiter, stStart);
		stEnd = stPosition != std::string::npos ? stPosition : strInput.length();

		std::string strElement = strInput.substr(stStart, stEnd - stStart);

		strElements.push_back(strtod(strElement.c_str(),NULL));

	}

	if (strInput.empty() || (strInput.substr(strInput.size() - strDelimiter.size(), strDelimiter.size()) == strDelimiter))
	{
		strElements.push_back(0.0);
	}

	return strElements;
}

void AINetTrainingData::closeTrainingDataFile(std::ifstream &ptrDataFile)
{
	/** This function is used to close the training data file after reading.
	 * \param ptrDataFile is a reference to the data file currently used.
	*/
	if (ptrDataFile.is_open()) 
	{
		// good, file is open.
		ptrDataFile.close();
	}
	else
	{
		std::cerr << "training data file could not be closed. ";
	}
}

std::string AINetTrainingData::convertFromCSVGermanStyle(std::string & strFileContents)
{
	/** this converts a german style *.csv file to a standard *.csv file
		\param strFileContents is a reference to the string data to be converted
		\return the value of the converted string
	*/
	if (this->bOptionCSVGER)
	{
		CodeFromWeb::ReplaceAllStrings(strFileContents, ",", ".");
		CodeFromWeb::ReplaceAllStrings(strFileContents, ";", ",");
	}
	return strFileContents;
}

std::string AINetTrainingData::convertToCSVStandardStyle(std::string & strFileContents)
{
	/** this converts a german style *.csv file to a standard *.csv file
		\param strFileContents is a reference to the string data to be converted
		\return the value of the converted string
	*/
	
	CodeFromWeb::ReplaceAllStrings(strFileContents, ",", ";");
	CodeFromWeb::ReplaceAllStrings(strFileContents, ".", ",");

	return strFileContents;
}

size_t AINetTrainingData::loadTrainingDataFile()
{
	/** This function is used to load all the training data.
	  * \return the number of unreadable lines.
	*/
	std::string theLine = "no open file.";
	int theFirstElement = 0;
	unsigned int iNumberOfLines = 0;
	unsigned int iNumberOfFalseLines = 0;
	int iTimePreviousElements = 0;	// How many previous rows for calculation?

	std::ifstream theAIDataFile;

	this->openTrainingDataFile(theAIDataFile);

	// clear old training data
	this->vvTrainingDataMatrix.clear();

	if (theAIDataFile.is_open())
	{
		// Read First Line and store for Information.
		std::getline(theAIDataFile, theLine);
		this->strAIDataFileHeader = theLine;

		//read the second line and reconfigure network.
		std::getline(theAIDataFile, theLine);
		this->convertFromCSVGermanStyle(theLine);
		if ((theLine.find_first_of(",") == theLine.npos))
		{
			// this is no german style csv but german style is set
			// disabling option ger
			this->bOptionCSVGER = false;
			
			// converting line backwards
			this->convertToCSVStandardStyle(theLine);
			
		}
		this->vdNetworkTopology = this->splitStringToSizeT(theLine, ",");

		std::vector<double> vdLocalVector(1 + this->vdNetworkTopology.front() + this->vdNetworkTopology.back());
		std::string loadedNumber = "";
		std::string tmpIsTimeData = "";
		// now start looking for maxiterations in aidatafile
		// and setting maximum iterations
		std::getline(theAIDataFile, theLine);
		this->convertFromCSVGermanStyle(theLine);
		theFirstElement = 0;
		int currentElementCounter = 0;
		while (theLine.length() > 0)
		{
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();
			// read value
			// example for this line:
			// 1000,5,3
			// 1000 iterations, -5 5 line above current line are historic data, 3 columns are used for historic data (5x3=)15 additional input nodes added.
			switch (currentElementCounter)
			{
			case 0:
				// first one on this line is maxiterations
				this->intMaxIterations = atoi(theLine.substr(0, theFirstElement).c_str());
				break;
			case 1:
				//second element on this line is number of elements in timescale
				this->intTimePreviousRows = atoi(theLine.substr(0, theFirstElement).c_str());
				break;
			case 2:
				// third element is number of columns to be used of previous data
				this->intTimePrevNumberOfColumns = atoi(theLine.substr(0, theFirstElement).c_str());
				break;
			case 5:
				// Todo change this
				this->dPercentVerificationData = std::min(0.0,std::max(1.0,atof(theLine.substr(0, theFirstElement).c_str())));
				break;
			case 6:
				tmpIsTimeData = theLine.substr(0, theFirstElement);
				std::transform(tmpIsTimeData.begin(), tmpIsTimeData.end(), tmpIsTimeData.begin(), ::tolower); 
				tmpIsTimeData.erase(tmpIsTimeData.find_last_not_of(" \n\r\t") + 1);
				if (tmpIsTimeData.compare("datetime") == 0)	this->intTimeDataMode = 4;
				else if (tmpIsTimeData.compare("date") == 0) this->intTimeDataMode = 2;
				else if (tmpIsTimeData.compare("time") == 0) this->intTimeDataMode = 3;
				else this->intTimeDataMode = 1;
				break;
			default:
				break;
			}
			// delete value from whole string
			theLine.erase(0, theFirstElement + 1);
			currentElementCounter += 1; // increase element counter
		}

		// one line for headers 
		std::getline(theAIDataFile, theLine);
		this->convertFromCSVGermanStyle(theLine);
		theFirstElement = 0;
		this->vStrTrainingDataColumns.push_back("intentionally left blank");
		while (theLine.length() > 0)
		{
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();
			// read value
			this->vStrTrainingDataColumns.push_back(theLine.substr(0, theFirstElement));
			// delete value from whole string
			theLine.erase(0, theFirstElement + 1);
		}

		//resize the vector
		vdLocalVector.clear();
		vdLocalVector.resize(1 + this->vdNetworkTopology.front() + this->vdNetworkTopology.back());
		this->intLinesRead = 0;

		while (!theAIDataFile.eof())
		{
			// now begin to read data values
			std::getline(theAIDataFile, theLine);
			++this->intLinesRead;
			this->convertFromCSVGermanStyle(theLine);
			// clear vector
			vdLocalVector.clear();
			theFirstElement = 0;
			vdLocalVector.push_back(1.0); // first element is base/threshold value and always set to 1.0

			// looking for first element
			if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
			else theFirstElement = (int)theLine.length();

			// clearing data from previous line
			loadedNumber = "";
			while ((theFirstElement > 0) && (vdLocalVector.size() < vdLocalVector.capacity())) // cancel if vector already has aincNetwork.NUMINPUTNODES() + aincNetwork.NUMOUTPUTNODES() +1 values
			{
				// read value
				loadedNumber = theLine.substr(0, theFirstElement);
				// delete value from whole string
				theLine.erase(0, theFirstElement + 1);
				vdLocalVector.push_back(strtod(loadedNumber.c_str(), NULL));

				// check for next column
				if (theLine.find_first_of(",") != theLine.npos) theFirstElement = (int)theLine.find_first_of(",");
				else theFirstElement = (int)theLine.length();
			}

			if (vdLocalVector.size() >= 1 + this->vdNetworkTopology.front() + this->vdNetworkTopology.back())
			{
				// counting number of lines and copying whole row to vector<vector>
				this->vvTrainingDataMatrix.push_back(vdLocalVector);
				iNumberOfLines += 1;
			}
			else
			{
				// counting false/erronous lines
				iNumberOfFalseLines += 1;
			}
		}
	}

	this->closeTrainingDataFile(theAIDataFile);

	return iNumberOfFalseLines;
}

bool AINetTrainingData::openTrainingDataFile(std::ifstream &ptrDataFile)
{
	/** This function is used to open the training data file
	* @param ptrDataFile is a reference to the training data *.csv file.
	  \returns true if successful.
		false if failed.
	*/
	ptrDataFile.open(this->strAIDataFileName.c_str());
	if (ptrDataFile.is_open())
	{
		// good file is open.
		return true;
	}
	else
	{
		std::cerr << "File Could not be opened. FILENAME:" << this->strAIDataFileName.c_str() << "\n";
		return false;
	}
}

size_t AINetTrainingData::loadTrainingData(std::string strFileName, bool bSample)
{
	/** This function is used to load training data from an *.csv file
		\param strFileName is the name of the file to be used.
		\param bSample If set to true sample data will be used. No data will be loaded.
	*/
	size_t retVal=this->intLinesRead;
	this->strAIDataFileName = strFileName;
	if (!bSample)
	{
		retVal= this->loadTrainingDataFile();
	}
	return retVal;
}

std::string AINetTrainingData::TrainingDataColumnName(size_t tmpColumn, bool shortList)
{
	/** This function will return the name of a given TrainingDataColumn
		\param tmpColumn the column which name is to be returned
		\param shortList if set to true it will directly access the given column
	*/

	std::string tmpString;
	if (!shortList)
	{
		// TODO uncomment and repair this function when done set shortlist to false
		/*// return the Name of the DataColumn
		std::string tmpString = "no column name";
		std::vector<unsigned int> retPullList;
		if (!shortList)
		{
			// generate List 
			if (this->iTimeNumInputColumns == 0)
			{
				// repeated inputs
				retPullList.resize(1 + this->getNumberOfInputNodes() + this->getNumberOfOutputNodes(), 0);
				for (unsigned int i = 0; i < retPullList.size(); i++)
				{
					if (i <= this->iNumRealInputNodes)
					{
						retPullList.at(i) = i;
					}
					else
					{
						retPullList.at(i) = i % this->iNumRealInputNodes + 1;
					}
				}
			}
			else
			{
				// repeated inputs
				retPullList.resize(1 + this->getNumberOfInputNodes() + this->getNumberOfOutputNodes(), 0);
				for (unsigned int i = 0; i < retPullList.size(); i++)
				{
					if (i <= this->iNumRealInputNodes)
					{
						retPullList.at(i) = i;
					}
					else
					{
						if ((i - this->iNumRealInputNodes) % (this->iTimeNumInputColumns) == 0)
						{
							retPullList.at(i) = this->iTimeNumInputColumns;
						}
						else
						{
							retPullList.at(i) = (i - this->iNumRealInputNodes) % (this->iTimeNumInputColumns);
						}
					}
				}
			}
			tmpColumn = retPullList.at(tmpColumn);
		}*/
	}
	else
	{
		if ((tmpColumn >= 0) && (tmpColumn < vStrTrainingDataColumns.size()))
		{
			// read a value if it is within valid range.
			tmpString = this->vStrTrainingDataColumns.at(tmpColumn);
		}
		else
		{
			tmpString = "unknown column";
		}
	}
	return tmpString;
}