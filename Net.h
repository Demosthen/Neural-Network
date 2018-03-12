#pragma once
#include "InLayer.h"
#include "ConvLayer.h"
#include "OutLayer.h"
#include "MaxPool.h"
using namespace std;
class Net
{
public:
	vector<shared_ptr<Layer>> layers;
	double learnRate;
	double beta1=-9999;
	double beta2=-9999;
	double stepSize = 0;
	double dropOut = 0;
	shared_ptr<pair<double, pair<double, pair<double, double> > > > bounds;
	int hidActFunct = 0;//0none,1sigmoid,2leakyrelu
	int outActFunct = 0;//0none,1sigmoid,2leakyrelu
	Vector<shared_ptr<Neuron> > sub;
	double threshold;//for gradient clipping
	Net();
	Net(double mlearnRate, vector<int>& numNeurons,double mthreshold, int mactFunct, int moutActFunct);
	Net(double mlearnRate, double mthreshold, double mbeta1, double mbeta2, double mdropOut, vector<shared_ptr<Layer>>& mlayers);
	Net(double mlearnRate, vector<int>& numNeurons, double mthreshold, double mbeta1, double mbeta2, double mdropOut, int mactFunct, int moutActFunct);
	void setTarget(vector<double>& targets);
	void backProp(vector<double> &targets);
	void update(double total);
	void testGrad(Vector<double>& input,vector<double>&targ);
	void feedForward(Vector<double>& input,bool eval=false);
	void printOut();
	void printState();
	void printErr();
	double getErr();
	double eval(vector<Vector<double>>& inputs, vector<vector<double>>& targs, double threshold);
	void printLayers();
	void saveNet(string fileName);
	~Net();
};

