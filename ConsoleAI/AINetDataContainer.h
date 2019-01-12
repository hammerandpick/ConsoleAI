#pragma once

class AINetDataContainer
{
public:
	AINetDataContainer();
	AINetDataContainer(double tdm);
	~AINetDataContainer();
	std::vector<std::vector<double>> vvTrainingDataMatrix = { {0.0} }; // training data from file loaded into this matrix; // this should be moved to a new class. reducing memory size
	std::vector<unsigned int> vdNetworkTopology = { 0 };
};

