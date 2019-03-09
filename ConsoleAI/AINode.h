#pragma once
class AINode
{
public:
	AINode();
	~AINode();
	bool setFunction(size_t iFunction);
	double getValue();
	double getError();
	double setValue(double dSetValue);
private:
	double dValue = 0.0;
	double dTargetValue = 0.0;
	double dThreshold = 0.0;
	double dError = 0.0;
	double dOutput = 0.0;
	size_t iNodeFunction = 0;
};

