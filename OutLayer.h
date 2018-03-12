#pragma once
#include "Layer.h"
class OutLayer :
	public Layer
{
public:
	OutLayer();
	OutLayer(int mprevNumNeur, int mnumNeur, int mactFunct);
	bool virtual eval( double margin);
	void virtual backProp(Vector<shared_ptr<Neuron> >& prevLayer, Vector<shared_ptr<Neuron> >&nextLayer);
	void virtual feedForward(Vector<shared_ptr<Neuron> >& prevLayer, double dropOut);
	double inline  virtual totalError();
	~OutLayer();
};

