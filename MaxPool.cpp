#include "stdafx.h"
#include "MaxPool.h"


MaxPool::MaxPool()
{
}
MaxPool::MaxPool(int Field, int numPrevX, int numPrevY, int numPrevZ)
{
	field = Field;
	neurons.reserve(ceil(((double)numPrevX) / Field)*ceil(((double)numPrevY) / Field)*numPrevZ);
	neurons.x = ceil(((double)numPrevX) / Field);
	neurons.y = ceil(((double)numPrevY) / Field);
	neurons.z = numPrevZ;
	deltas.resize(numPrevX*numPrevY*numPrevZ, 0);
	locs.resize(ceil(((double)numPrevX) / Field)*ceil(((double)numPrevY) / Field)*numPrevZ);
	locs.x = ceil(((double)numPrevX) / Field);
	locs.y = ceil(((double)numPrevY) / Field);
	locs.z = numPrevZ ;
	for (int i = 0; i < ceil(((double)numPrevX) / Field)*ceil(((double)numPrevY) / Field)*numPrevZ; i++) {
		neurons.push_back(shared_ptr<Neuron>(new Neuron()) );
	}
}
void MaxPool::feedForward(Vector<shared_ptr<Neuron> > &input, double dropOut, bool eval) {
	vector<bool> toSave;
	toSave.resize(neurons.size(), false);
	for (int a = 0; a < input.z; a++) {
		for (int i = 0; i < input.x; i += field) {
			for (int j = 0; j < input.y; j += field) {
				double maxVal = -1526;
				pair<int, int> coord = { -500, -500};
				for (int u = 0; u < field; u++) {
					for (int g = 0; g < field; g++) {
						if (Helper::checkBounds(shared_ptr<ppp>(new ppp( 0,{0,{input.x,input.y}} )), { u + i,g + j }) && maxVal < input(i + u, j + g, a)->output) {
							maxVal = input(i + u, j + g, a)->output;
							coord.first = u;
							coord.second = g;
						}
					}
				}

				neurons(i / field, j / field, a)->output = maxVal;
				locs(i / field, j / field, a) = { i+coord.first,{j+coord.second,a} };
			}

		}
	}
}
void MaxPool::backProp(Vector<shared_ptr<Neuron>> &prevLayer, Vector<shared_ptr<Neuron>>&nextLayer) {
	for (int i = 0; i < neurons.size(); i++) {
		prevLayer(locs(i).first, locs(i).second.first, locs(i).second.second)->delta = neurons(i)->delta;
		neurons(i)->delta = 0;
	}
}
MaxPool::~MaxPool()
{
}
