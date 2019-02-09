#include "pch.h"
#include "stdafx.h"
#include <vector>
#include "AINetDataContainer.h"


AINetDataContainer::AINetDataContainer()
{
	/** Constructior for AINetDataContainer
	  * @param tdm -- double optional parameter, can be used for initalizing the whole network
	*/

	this->vvTrainingDataMatrix.clear();
	this->vdNetworkTopology.clear();
}

AINetDataContainer::AINetDataContainer(double tdm)
{
	this->vvTrainingDataMatrix = { {tdm} };
	this->vdNetworkTopology = { 0 };
}

AINetDataContainer::~AINetDataContainer()
{
	this->vvTrainingDataMatrix.clear();
	this->vdNetworkTopology.clear();
}
