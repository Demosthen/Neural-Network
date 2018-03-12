#include "stdafx.h"
#include "Layer.h"
Layer::Layer()
{
	
}
Layer::Layer(int mprevNumNeur, int mnumNeur, int mactFunct)
{	
	actFunct = mactFunct;
	prevNumNeur = mprevNumNeur;
	numNeur = mnumNeur;
	neurons.reserve(numNeur);
	for (int i = 0; i < numNeur; i++) {
		neurons.push_back(shared_ptr<Neuron>(new Neuron(prevNumNeur, mactFunct, { -1,{ -1,{ -1,-1 } } })));
	}
	neurons.x = numNeur;
	
}

void Layer::backProp(Vector<shared_ptr<Neuron>> &prevLayer, Vector<shared_ptr<Neuron>>&nextLayer ) {
	//double tempSum = 0;
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->delta*=neurons(i)->actDeriv();
		//delta = min(threshold, abs(delta))*(2 * signbit(delta) - 1)*-1;
		neurons(i)->bGradient += neurons(i)->delta;

		for (int j = 0; j < prevLayer.size(); j++) {
			//tempSum += delta*neurons[i].weight[j];
			*neurons(i)->gradient(j) += neurons(i)->delta*prevLayer(j)->output;
			prevLayer(j)->delta += neurons(i)->delta* *neurons(i)->weight(j);
		}
		neurons(i)->delta = 0;
	}
}

void Layer::feedForward(Vector<shared_ptr<Neuron> > &prevLayer, double dropOut, bool eval) {
		for (int i = 0; i < neurons.size(); i++) {
			neurons(i)->feedForward(prevLayer, dropOut,eval);
		}
}
void Layer::feedForward(Vector<double>& input, bool eval) {
}
void Layer::feedForward(Vector<double> &prevLayer, double dropOut,bool eval) {
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->feedForward(prevLayer, dropOut,eval);
	}
}
void Layer::printOut() {
	for (int i = 0; i < neurons.size(); i++) {
		cout << neurons(i)->output << endl;
	}
}
void Layer::printState() {
	cout << "NEWLAYER" << endl;
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->printState();
	}
}
double Layer::totalError() { return 0; }

void Layer::update(double learnRate) {//plain sgd
	for (int i = 0; i < neurons.size(); i++) {
		for (int j = 0; j < neurons(i)->weight.size(); j++) {
				*neurons(i)->weight(j) -= learnRate* *neurons(i)->gradient(j);
				*neurons(i)->gradient(j) = 0;
		}
		*neurons(i)->bias -= learnRate*neurons(i)->bGradient;
		neurons(i)->bGradient = 0;
	}
}
void Layer::update(double learnRate, double beta1, double beta2, double total) {//Adam
	
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->update(learnRate, beta1, beta2, total);
	}
}
Layer::~Layer()
{
}
