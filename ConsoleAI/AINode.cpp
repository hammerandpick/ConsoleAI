#include "pch.h"
#include "AINode.h"


AINode::AINode()
{
}


AINode::~AINode()
{
}

bool AINode::setFunction(size_t iFunction)
{
	/** This function is used to set the calculation function of this node.
		\param iFunction This is the type of the calculation function
	*/
	this->iNodeFunction = iFunction;
	return false;
}

double AINode::getValue()
{
	/** Return the current Value of this node 
		\return The value of this node */
	return this->dValue;
}

double AINode::getError()
{
	/** Return the error value at this node.
		\return The error value of this node.
	*/
	return this->dTargetValue - this->dValue;
}

double AINode::setValue(double dSetValue)
{
	/** Set the value of this node.
		\param dSetValue The value of this node.*/
	this->dValue = dSetValue;
	return this->dValue;
}
