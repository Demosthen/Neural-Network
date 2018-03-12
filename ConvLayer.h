#pragma once
#include "Filter.h"
#include "Layer.h"
class ConvLayer
	:public Layer
{
	int width;
	int height;
	pair<int,int>  imgDims;//width,depth height=width
	int recField;
	int depth;
	int stride = 1;
	int arr1[4] = { 0,1,0, 1 };
	int arr2[4] = { 0,0,1,1 };
	int numKernels = 0;
	int numFilters = 0;
	int actFunct;
	bool pool;
public:
	ConvLayer();
	ConvLayer(pair<int, int>& dims, int rField, int stepSize, int mactFunct, bool Pool, int mnumKernels, int mdepth);
	ConvLayer(pair<int, int>& dims, int rField, int stepSize, int mactFunct, bool Pool, int mnumKernels,shared_ptr<ppp> &x, int mdepth);
	void maxPool();
	void virtual inline feedForward(Vector<double>& input, bool eval=false);
	double getDelta(Vector<shared_ptr<Neuron> >& nextLayer, int neur);
	void backProp(Vector<shared_ptr<Neuron>>& prevLayer, Vector<shared_ptr<Neuron>>& nextLayer);
	~ConvLayer();
};

