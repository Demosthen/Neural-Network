
#pragma once

#include "Vector.h"
#include "Helper.h"

using namespace std;
#ifndef ppp 
#define ppp pair<double,pair<double, pair<double,double> > >;
#endif
class Neuron
{
public:
	//vector<vector<double> > subWeight;
	//vector<vector<double> > &weight;
	Vector<double> subWeight;
	double epsilon = 0.00000001;
	shared_ptr<ppp> bounds;
	Vector<shared_ptr<double> > weight;
	double output = 0;
	double delta;
	//vector<vector<double> > subGrad;
	//vector<vector<double> > &gradient;
	Vector<double> subGrad;
	Vector<shared_ptr<double> > gradient;
	double subBGrad=0;
	double bGradient;
	pair<int, int> filterSize = { -1,-1 };//x,y
	int filterDepth=1;
	//vector<vector<double> > subM1;
	//vector<vector<double> > &m1;
	Vector<double> subM1;
	Vector<shared_ptr<double> > m1;
	int actFunct = 0;//0none,1sigmoid,2leakyrelu
	double subBM1;
	double bm1;
	double threshold;
	//vector<vector<double> > subV1;
	//vector<vector<double> > &v1;
	Vector<double> subV1;
	Vector<shared_ptr<double> > v1;
	double subBV1 = 0;
	double bv1;
	double subBias=0;
	shared_ptr<double> bias;
	Vector<double> mInputs;
	pair<int, pair<int, pair<int, int> > >  pos = { -1,{-1,{-1,-1}} };//3rd val does nothing
	const double E = 2.71828;
	Vector<pair<int, pair<int, int> > > loc;
	Neuron();
	Neuron(int numPrev, int mactFunct, pair<int, pair<int,pair<int,int > > > Pos);
	/*Neuron(vector<vector<double>>& w, double & b, vector<vector<double>>& grad, double & bGrad,
		vector<vector<double>>& M1, vector<vector<double>>& V1, double & bM1, double & bV1, int mactFunct, pair<int, int>Pos = { -1,-1 });*/
	Neuron(Vector<shared_ptr<double>>& w, double & b, Vector<shared_ptr<double>>& grad, double & bGrad, Vector<shared_ptr<double>>& M1, 
		Vector<shared_ptr<double>>& V1, double bM1, double bV1, int mactFunct, pair<int, pair<int, pair<int, int>>> Pos);
	Neuron(shared_ptr<Neuron> a, pair<int, pair<int, pair<int, int> > > Pos);
	Neuron(int mactFunct, pair<int, int>& mfilterSize, pair<int, pair<int, pair<int, int>>> Pos, int fDepth);
	Neuron(shared_ptr<Neuron> a, pair<int, pair<int, pair<int, int> > > Pos,shared_ptr<ppp>& x);
	Neuron(int mactFunct, pair<int, int>& mfilterSize, pair<int, pair<int, pair<int, int>>> Pos, int fDepth,shared_ptr<ppp>& x);
	double inline sigmoid(double netInput);
	double inline leakyReLu(double netIn);
	double activation(double netIn);
	double calcError(double target);
	double sigPrime(double out);
	double errPrime(double target);
	double inline leakyReLuDeriv(double out);
	double inline actDeriv(double out);
	double actDeriv();
	double feedForward(Vector<double> &inputs, double dropOut, bool eval=false) {
		double sum = 0;
		if (((double)rand()) / RAND_MAX < dropOut&&!eval) {
			output = 0;
			return 0;
		}
		if (filterSize.first!=-1) {
			for (int i = 0; i < filterSize.first; i++) {
				for (int j = 0; j < filterSize.second; j++) {
					for (int k = 0; k < filterDepth; k++) {
						int p1 = pos.first + i+bounds->first;
						int p2 = pos.second.first + j+bounds->second.first;
						if (Helper::checkBounds(bounds, make_pair( p1,p2 ))) {
							sum += inputs(p1, p2, k)* *weight(i, j,k);
							mInputs(i, j, k) = inputs(p1, p2, k);
							loc(i, j, k) = { p1,{ p2,k } };
						}
					}
				}
			}
		}
		else {
			for (int i = 0; i < inputs.size(); i++) {
				sum += inputs(i) * *weight(i);
			}
		}
		sum += *bias;
		output = activation(sum);
		return output;
	}
	double feedForward(Vector<shared_ptr<Neuron>> &inputs, double dropOut, bool eval=false) {
		double sum = 0;
		if (((double)rand()) / RAND_MAX < dropOut&&!eval) {
			output = 0;
			return 0;
		}
		if (filterSize.first != -1) {
			for (int i = 0; i < filterSize.first; i++) {
				for (int j = 0; j < filterSize.second; j++) {
					for (int k = 0; k < filterDepth; k++) {
						int p1 = bounds->first + pos.first + i;
						int p2 = bounds->second.first + pos.second.first + j;
						if (Helper::checkBounds(bounds, { p1,p2 })) {
							sum += inputs(p1, p2, k)->output* *weight(i, j, k);
							mInputs(i, j, k) = inputs(p1, p2, k)->output;
							loc(i, j, k) = { (pos.first+i),{ pos.second.first+j,k } };
						}
					}
				}
			}
		}
		else {
			for (int i = 0; i < inputs.size(); i++) {
				sum += inputs(i)->output * *weight(i);
			}
		}
		sum += *bias;
		output = activation(sum);
		return output;
	}
	double operator*(double &x);
	virtual void inline operator=(Neuron &x);
	double feedForward();
	void printState();
	void update(double learnRate, double beta1, double beta2, double total);
	~Neuron();
};

