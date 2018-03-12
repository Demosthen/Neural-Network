#pragma once
#include "Neuron.h"
using namespace std;

class Layer
{

public:
	Vector<shared_ptr<Neuron>> neurons;
	int prevNumNeur;
	int numNeur;
	shared_ptr<pair<double, pair<double, pair<double, double> > > > bounds;
	int actFunct = 0;//0none,1sigmoid,2leakyrelu
	double threshold;
	double epsilon = 0.0000001;
	vector<double> targets;
	Vector<double> deltas;
	Vector<pair<int, pair<int, int> >> locs;
	string type = "Layer";
	Layer();
	Layer(int mprevNumNeur, int mnumNeur, int mactFunct);
	double inline getDelta(Vector<shared_ptr<Neuron> > &nextLayer, int neur);
	void virtual backProp(Vector<shared_ptr<Neuron> > &prevLayer, Vector<shared_ptr<Neuron>>&nextLayer);
	void virtual feedForward(Vector<shared_ptr<Neuron> >& prevLayer, double dropOut,bool eval=false);
	void virtual feedForward(Vector<double>& prevLayer, double dropOut,bool eval=false);
	void virtual feedForward(Vector<double>& input, bool eval=false);
	double virtual totalError();
	bool virtual eval(double margin) { return 0; }
	void printOut();
	void printState();
	void virtual update(double learnRate);
	void virtual update(double learnRate, double beta1, double beta2, double total);
	~Layer();
};

