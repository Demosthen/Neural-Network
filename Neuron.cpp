#include "stdafx.h"
#include "Neuron.h"
using namespace std;
Neuron::Neuron(){}

template<class type>
void inline zeroInit(Vector<shared_ptr<type> > &vec) {
	for (int i = 0; i < vec.size(); i++) {
		vec(i) = shared_ptr<double>(new double(0));
	}
}
Neuron::Neuron(int numPrev, int mactFunct, pair<int, pair<int, pair<int,int> > > Pos = { -1,{-1,{-1,-1}} })
{	
	actFunct = mactFunct;
	pos = Pos;
	weight.resize(numPrev,0);
	weight.x = numPrev;
	for (int i = 0; i < numPrev; i++) {
		weight(i) = shared_ptr<double>(new double((((double)rand()) / (100 * RAND_MAX))));
	}
	bias = shared_ptr<double>(new double(((double)rand()) / (100 * RAND_MAX)));
	bGradient = 0;
	gradient.resize(numPrev);
	gradient.x = numPrev;
	zeroInit(gradient);
	m1.resize(numPrev);
	m1.x = numPrev;
	v1.resize(numPrev);
	v1.x = numPrev;
	zeroInit(m1);
	zeroInit(v1);
	if (pos.first != -1) {
		mInputs.resize(filterSize.first*filterSize.second*filterDepth,1);
		mInputs.x = filterSize.first;
		mInputs.y = filterSize.second;
		mInputs.z = filterDepth;
		loc.resize(filterSize.first*filterSize.second*filterDepth);
		loc.x = filterSize.first;
		loc.y = filterSize.second;
		loc.z = filterDepth;
	}
}
Neuron::Neuron(Vector<shared_ptr<double> >& w, double & b, Vector<shared_ptr<double> >& grad, double & bGrad, Vector<shared_ptr<double> >& M1, 
	Vector<shared_ptr<double> >& V1, double bM1, double bV1, int mactFunct, pair<int, pair<int, pair<int,int> > >Pos) 
	:weight(w), gradient(grad), bGradient(bGrad), m1(M1), v1(V1), bm1(bM1), bv1(bV1)
{
	bias = shared_ptr <double>(new double(b));
	actFunct = mactFunct;
	pos = Pos;
	//bounds not set
	if (pos.first != -1) {
		mInputs.resize(filterSize.first*filterSize.second*filterDepth,0);
		mInputs.x = filterSize.first;
		mInputs.y = filterSize.second;
		mInputs.z = filterDepth;
		loc.resize(filterSize.first*filterSize.second*filterDepth);
		loc.x = filterSize.first;
		loc.y = filterSize.second;
		loc.z = filterDepth;
	}
}
Neuron::Neuron(shared_ptr<Neuron> a, pair<int, pair<int, pair<int,int> > > Pos ) : weight(a->weight),filterSize(a->filterSize),filterDepth(a->filterDepth),bias(a->bias) {
	cout << "neurConstruct" << endl;
	pos = Pos;
	gradient.resize(filterSize.first*filterSize.second*filterDepth);
	gradient.x = filterSize.first;
	gradient.y = filterSize.second;
	gradient.z = filterDepth;
	zeroInit(gradient);
	m1.resize(filterSize.first*filterSize.second*filterDepth);
	m1.x = filterSize.first;
	m1.y = filterSize.second;
	m1.z = filterDepth;
	zeroInit(m1);
	v1.resize(filterSize.first*filterSize.second*filterDepth);
	v1.x = filterSize.first;
	v1.y = filterSize.second;
	v1.z = filterDepth;
	zeroInit(v1);
	mInputs.resize(filterSize.first*filterSize.second*filterDepth,1);
	mInputs.x = filterSize.first;
	mInputs.y = filterSize.second;
	mInputs.z = filterDepth;

	loc.resize(filterSize.first*filterSize.second*filterDepth);
	loc.x = filterSize.first;
	loc.y = filterSize.second;
	loc.z = filterDepth;
	//bounds not set
	//*bias = ((double)rand()) / (100 * RAND_MAX);
	//bias = 0;
	bGradient = 0;
}
Neuron::Neuron(int mactFunct, pair<int, int>& mfilterSize, pair<int, pair<int,pair<int,int> > > Pos, int fDepth) 
{
	cout << "neurConstruct" << endl;
	filterSize = mfilterSize;
	actFunct = mactFunct;
	filterDepth = fDepth;
	weight.resize(filterSize.first*filterSize.second*fDepth);
	weight.x = filterSize.first;
	weight.y = filterSize.second;
	weight.z = filterDepth;
	pos = Pos;
	gradient.resize(filterSize.first*filterSize.second*fDepth);
	gradient.x = filterSize.first;
	gradient.y = filterSize.second;
	gradient.z = filterDepth;
	zeroInit(gradient);
	m1.resize(filterSize.first*filterSize.second*fDepth);
	m1.x = filterSize.first;
	m1.y = filterSize.second;
	m1.z = filterDepth;
	zeroInit(m1);
	v1.resize(filterSize.first*filterSize.second*fDepth);
	v1.x = filterSize.first;
	v1.y = filterSize.second;
	v1.z = filterDepth;
	zeroInit(v1);
	mInputs.resize(filterSize.first*filterSize.second*fDepth,0);
	mInputs.x = filterSize.first;
	mInputs.y = filterSize.second;
	mInputs.z = fDepth;
	loc.resize(filterSize.first*filterSize.second*fDepth);
	loc.x = filterSize.first;
	loc.y = filterSize.second;
	loc.z = fDepth;
	//bounds not set
	cout << "preWeight" << endl;
	for (int i = 0; i < filterSize.first; i++) {
		for (int j = 0; j < filterSize.second; j++) {
			for (int k = 0; k < filterDepth; k++) {
				weight(i, j,k) = shared_ptr<double>(new double(((double)rand()) / (100 * RAND_MAX)));
				//weight(i, j,k) = shared_ptr<double>(new double(1));
			}
		}
	}
	cout << "weightsAssigned" << endl;
	bias = shared_ptr<double>(new double(((double)rand()) / (100 * RAND_MAX)));
	//bias = 0;
	bGradient = 0;
}
Neuron::Neuron(shared_ptr<Neuron> a, pair<int, pair<int, pair<int, int>>> Pos, shared_ptr<ppp>& x):Neuron(a,pos)
{
	pos = Pos;
	bounds = x;
}
Neuron::Neuron(int mactFunct, pair<int, int>& mfilterSize, pair<int, pair<int, pair<int, int>>> Pos, int fDepth, shared_ptr<ppp>& x)
:Neuron(mactFunct, mfilterSize,Pos,fDepth){
	pos = Pos;
	bounds = x;
}
double inline Neuron::sigmoid(double netInput) {//tested
	return 1.00000000 / (1.00000000 + pow(E, -netInput));
}
double inline Neuron::leakyReLu(double netIn) {
	//Leaky ReLu
	if (netIn <= 0) {
		return 0.01*netIn;
	}
	else {
		return netIn;
	}
}
double Neuron::activation(double netIn) {
	switch (actFunct) {
	case 0:
		return netIn;
	case 1:
		return sigmoid(netIn);
	case 2:
		return leakyReLu(netIn);
	}
}
double Neuron::calcError(double target) {
	double err = 0;
	double out = output;
	err = (target - out)*(target - out) / 2;
	return err;
}
double Neuron::sigPrime(double out) {

	return out*(1 - out);
}
double Neuron::errPrime(double target) {
	
	return -(target - output);
}
double inline Neuron::leakyReLuDeriv(double out) {
	if (out <= 0) {
		return 0.01;
	}
	else {
		return 1;
	}
}
double inline Neuron::actDeriv(double out) {
	switch (actFunct) {
	case 0:
		return 1;
	case 1:
		return sigPrime(out);
	case 2:
		return leakyReLuDeriv(out);
	}
}
double Neuron::actDeriv() {
	switch (actFunct) {
	case 0:
		return 1;
	case 1:
		return sigPrime(output);
	case 2:
		return leakyReLuDeriv(output);
	}
}


double Neuron::operator* (double &x) {
	return output*x;
}
void inline Neuron::operator=(Neuron & x)
{
	*this = x;
}
double inline Neuron::feedForward() {
	return output;
}
void Neuron::printState() {
	cout << "b" << bias << endl;
	for (int i = 0; i < weight.size(); i++) {
		cout << weight(i) << endl;
	}
}
void Neuron::update(double learnRate, double beta1, double beta2, double total) {
	if (beta1 != -9999) {
		double eb1 = pow(beta1, total);
		double eb2 = pow(beta2, total);
		
		for (int j = 0; j < weight.size(); j++) {
			assert(*gradient(j) != 0);
				*m1(j) = beta1* *m1(j) + *gradient(j) * (1 - beta1);
				*v1(j) = beta2* *v1(j) + *gradient(j) * *gradient(j) * (1 - beta1);
				*weight(j) -= learnRate * *m1(j) / ((1 - eb1)*sqrt(*v1(j) / (1 - eb2)) + epsilon);
				*gradient(j) = 0;
			
		}
		bm1 = beta1 * bm1 + bGradient * (1 - beta1);
		bv1 = beta2 * bv1 + bGradient*bGradient * (1 - beta2);

		*bias -= learnRate * bm1 / ((1 - eb1)*sqrt(bv1 / (1 - eb2)) + epsilon);
		bGradient = 0;
	}
	else {
		for (int i = 0; i < weight.size(); i++) {
			*weight(i) -= learnRate* *gradient(i);
			*gradient(i) = 0;
		}
		*bias -= learnRate*bGradient;
		bGradient = 0;
	}
}
Neuron::~Neuron()
{
}
