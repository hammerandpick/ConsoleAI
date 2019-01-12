#include "pch.h"
#include<vector>
#include "AINetDataContainer.h"


AINetDataContainer::AINetDataContainer()
{
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
