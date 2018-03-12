#include "stdafx.h"
#include "ConvLayer.h"

ConvLayer::ConvLayer() {

}
ConvLayer::ConvLayer(pair<int, int>& dims, int rField, int stepSize, int mactFunct, bool Pool, int mnumKernels, int mdepth)
{
	cout << "CONVLAYERSTART" << endl;
	imgDims = dims;
	type = "ConvLayer";
	actFunct = mactFunct;
	recField = rField;
	pool = Pool;
	depth = mdepth;
	stride = stepSize;
	width = (imgDims.first-rField) / stride +1;
	height = (imgDims.second - rField) / stride + 1;
	numKernels = mnumKernels;
	deltas.resize(imgDims.first*imgDims.second);
	deltas.x = imgDims.first;
	deltas.y = imgDims.second;
	neurons = Vector<shared_ptr<Neuron>>(width,height,numKernels);
	neurons.x = width;
	neurons.y = height;
	neurons.z = numKernels;
	bounds = shared_ptr<ppp>(new ppp(0, { 0,{imgDims.first,imgDims.second} }));
	//cout << "vec "<<neurons.size()<< endl;
	for (int a = 0; a < numKernels; a++) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (i == 0 && j == 0) {
					neurons.assign(i, j, a, shared_ptr<Neuron>(new Neuron(actFunct, make_pair(rField, rField), { i*stride,{ j*stride,{ 0,a } } },depth,bounds)));
					
				}
				else if(Helper::checkBounds(bounds,{i*stride,j*stride})){
					neurons.assign(i, j, a, shared_ptr<Neuron>(new Neuron(neurons(0, 0, a), { i*stride,{ j*stride,{ 0,a } } },bounds)));
				}
			}
		}
	}
	cout << "";
}
ConvLayer::ConvLayer(pair<int, int>& dims, int rField, int stepSize, int mactFunct, bool Pool, int mnumKernels, shared_ptr<ppp>& x, int depth)
:ConvLayer(dims,rField,stepSize,mactFunct, Pool,mnumKernels, depth){
	*bounds = *x;
}
void ConvLayer::maxPool() {// 2*2
	vector<bool> toSave;
	toSave.resize(neurons.size(),false);
	for (int a = 0; a < neurons.z; a++) {
		for (int i = 0; i < neurons.x; i += 2) {
			for (int j = 0; j < neurons.y; j += 2) {
					double maxVal = -1526;
					pair<int, int> coord;
					for (int u = 0; u < sizeof(arr1)/sizeof(*arr1); u++) {
						for (int g = 0; g < sizeof(arr2)/sizeof(*arr2); g++) {
							if (Helper::checkBounds(bounds, { arr1[u] + i,arr2[g] + j })&&maxVal < neurons(i + arr1[u], j + arr2[g], a)->output) {
								maxVal = neurons(i + arr1[u], j + arr2[g],a)->output;
								//cout << maxVal <<" "<<i+arr1[u]<<" "<<j<<" "<<arr2[g]<< endl;
								coord.first = u;
								//cout << arr1[u]+i<<" "<<arr2[g]+j << endl;
								coord.second = g;
							}
						}
					}
					//cout << coord.first<<" "<<coord.second << endl;
					toSave[neurons.getIndex(i+arr1[coord.first], j+arr2[coord.second], a)]=true;

			}

		}
	}
	for (int i = 0; i < neurons.size(); i++) {
		if (!toSave[i]) {
			neurons(i)->output = 0;
		}
	}
}
void ConvLayer::feedForward(Vector<double> &input, bool eval) {
	for (int i = 0; i < neurons.size(); i++) {
		neurons(i)->feedForward(input,0);
	}
}
void ConvLayer::backProp(Vector<shared_ptr<Neuron>> &prevLayer, Vector<shared_ptr<Neuron>>&nextLayer) {
	//double tempSum = 0;
	bool l1 = prevLayer(0) == neurons(0);
	for (int i = 0; i < neurons.size(); i++) {
			neurons(i)->delta *=neurons(i)->actDeriv();//delta calced in nextlayer already
			//delta = min(threshold, abs(delta))*(2 * signbit(delta) - 1)*-1;
			neurons(i)->bGradient += neurons(i)->delta;
			for (int j = 0; j < neurons(i)->mInputs.size(); j++) {
				//tempSum += delta*neurons[i].weight[j];
				*neurons(i)->gradient(j) += neurons(i)->delta*neurons(i)->mInputs(j);
				if (!l1) {
					prevLayer(neurons(i)->loc(j).first, neurons(i)->loc(j).second.first, neurons(i)->loc(j).second.second)->delta+=neurons(i)->delta;
				}
			}

			neurons(i)->delta = 0;
	}
}

ConvLayer::~ConvLayer()
{
}
