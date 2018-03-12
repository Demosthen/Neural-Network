#pragma once
#include "Layer.h"
class InLayer :
	public Layer
{
public:
	InLayer();
	InLayer(int mnumNeur);
	void setInput(Vector<double>& input);
	double virtual backProp(double deltaSum, Vector<Neuron>& prevLayer);
	void virtual backProp(Vector<shared_ptr<Neuron> > &prevLayer, Vector<shared_ptr<Neuron>>&nextLayer);
	void virtual update(double learnRate);
	void virtual feedForward (Vector<double>& input);
	~InLayer();
};

