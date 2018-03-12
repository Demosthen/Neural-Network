#include "stdafx.h"
#include "InLayer.h"


InLayer::InLayer()
{
}
InLayer::InLayer(int mnumNeur) :Layer(0, mnumNeur,0)
{
	type = "InLayer";
}
void InLayer::setInput(Vector<double> &input) {
	neurons.x = input.x;
	neurons.y = input.y;
	neurons.z = input.z;
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->output = input(i);
	}
}
double InLayer::backProp(double deltaSum, Vector<Neuron> &prevLayer) {
	return deltaSum;
}

void InLayer::backProp(Vector<shared_ptr<Neuron>>& prevLayer, Vector<shared_ptr<Neuron>>& nextLayer)
{

}


void InLayer::update(double learnRate) {
}

void InLayer::feedForward(Vector<double> &input) {
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->output = input(i);
	}
}
InLayer::~InLayer()
{
}
