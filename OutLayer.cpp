#include "stdafx.h"
#include "OutLayer.h"


OutLayer::OutLayer()
{
}
OutLayer::OutLayer(int mprevNumNeur, int mnumNeur, int mactFunct):Layer(mprevNumNeur, mnumNeur,mactFunct)
{
	type = "OutLayer";
}
void OutLayer::feedForward(Vector<shared_ptr<Neuron>>&prevLayer, double dropOut) {
	Layer::feedForward(prevLayer, 0);
}
bool OutLayer::eval( double margin) {
	double maxOut = -1;
	int maxIndex = -1;
	int targ = -2;
	int cnt = 0;
	for (int i = 0; i < neurons.size(); i++) {
		//cout << neurons[i].output <<" "<<i<<" "<< endl;
		if (isnan(neurons(i)->output) || isinf(neurons(i)->output)) {
			
			return false;
		}
		if (maxOut < neurons(i)->output) {
			maxOut = neurons(i)->output;
			++cnt;
			//cout << cnt << endl;
			maxIndex = i;
		}
		if (targets[i] > 0.5) {
			targ = i;
		}
	}
	return maxIndex == targ;
}
void OutLayer::backProp(Vector<shared_ptr<Neuron> > &prevLayer, Vector<shared_ptr<Neuron> >&nextLayer) {
	
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->delta = neurons(i)->errPrime(targets[i])*neurons(i)->actDeriv();
		neurons(i)->bGradient += neurons(i)->delta;
		for (int j = 0; j < prevLayer.size(); j++) {
			//tempSum += delta*neurons[i].weight[j];
			*neurons(i)->gradient(j) += neurons(i)->delta*prevLayer(j)->output;
			prevLayer(j)->delta += neurons(i)->delta* *neurons(i)->weight(j);
		}

	}
}
double inline OutLayer::totalError() {
	double err = 0;
	for (int i = 0; i < neurons.size(); i++) {
		err += neurons(i)->calcError(targets[i]);
	}
	return err;
}
OutLayer::~OutLayer()
{
}
