#pragma once
#include "Layer.h"
class MaxPool :
	public Layer
{
public:
	int field;
	MaxPool();
	MaxPool(int Field, int numPrevX, int numPrevY, int numPrevZ);
	virtual void feedForward(Vector<shared_ptr<Neuron>>& input,double dropOut, bool eval=false);
	virtual void backProp(Vector<shared_ptr<Neuron>>& prevLayer, Vector<shared_ptr<Neuron>>& nextLayer);
	~MaxPool();
};

