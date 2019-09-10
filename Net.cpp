#include "stdafx.h"
#include "Net.h"



Net::Net()
{
}
Net::Net(double mlearnRate, vector<int> &numNeurons, double mthreshold, int mactFunct, int moutActFunct) {
	int numLayers = numNeurons.size();
	threshold = mthreshold;
	outActFunct = moutActFunct;
	hidActFunct = mactFunct;
	layers.resize(numLayers);
	layers[0]=shared_ptr<InLayer>(new InLayer(numNeurons[0]));
	layers[0]->threshold = threshold;
	learnRate = mlearnRate;
	for (int i = 1; i < numNeurons.size()-1; i++) {
		layers[i] =shared_ptr<Layer>(new Layer(numNeurons[i - 1], numNeurons[i],hidActFunct));
		layers[i]->threshold = threshold;
	}
	layers[numNeurons.size() - 1] = shared_ptr<OutLayer>(new OutLayer(numNeurons[numLayers - 2], numNeurons[numLayers - 1],outActFunct));
	layers[numNeurons.size() - 1]->threshold = threshold;
}
Net::Net(double mlearnRate, double mthreshold, double mbeta1, double mbeta2, double mdropOut, vector<shared_ptr<Layer> > &mlayers) {
	learnRate = mlearnRate;
	threshold = mthreshold;
	beta1 = mbeta1;
	beta2 = mbeta2;
	layers = mlayers;
}


Net::Net(double mlearnRate, vector<int> &numNeurons, double mthreshold, double mbeta1, double mbeta2, double mdropOut, int mactFunct, int moutActFunct): Net(mlearnRate,numNeurons,mthreshold,mactFunct,moutActFunct) {
	beta1 = mbeta1;
	beta2 = mbeta2;
	dropOut = mdropOut;
}
void Net::setTarget(vector<double> &targets) {
	layers[layers.size() - 1]->targets = targets;
}
void Net::backProp(vector<double> &targets) {
	layers[layers.size() - 1]->targets = targets;
	
	layers[layers.size() - 1]->backProp(layers[layers.size() - 2]->neurons,sub);
	for (int i = layers.size()-2; i >0; i--) {
		layers[i]->backProp(layers[i - 1]->neurons,layers[i+1]->neurons);
	}
	layers[0]->backProp(layers[0]->neurons, layers[1]->neurons);
}
void Net::update(double total) {//total number of cases seen
	for (int i = 0; i < layers.size(); i++) {
		if (beta1 != -9999) {
			layers[i]->update(learnRate, beta1, beta2, total);//add beta parameters for ADAM
		}
		else {
			layers[i]->update(learnRate);
		}
	}
}
void Net::testGrad(Vector<double>&input, vector<double>&targ) {
	setTarget(targ);
	double epsilon = 0.0000001;
	double maxDiff = 0;
	vector<vector<vector<double> > > theorGrads;
	theorGrads.resize(layers.size());
	cout << "theorGrads" << endl;
	for (int i = 1; i < layers.size(); i++) {
		theorGrads[i].resize(layers[i]->neurons.size());
		for (int j = 0; j < layers[i]->neurons.size(); j++) {
			theorGrads[i][j].resize(layers[i]->neurons(j)->weight.size());
			for (int k = 0; k < layers[i]->neurons(j)->weight.size(); k++) {
				*layers[i]->neurons(j)->weight(k) += epsilon;
				feedForward(input);
				double err1 = getErr();
				*layers[i]->neurons(j)->weight(k) -= 2*epsilon;
				feedForward(input);
				double err2 = getErr();
				theorGrads[i][j][k]=((err1 - err2) / (2 * epsilon));
			}
		}
	}
	cout << "preback" << endl;
	backProp(targ);
	cout << "doneback" << endl;
	double avgDiff = 0;
	double cnt = 0;
	double denom1 = 0;
	double denom2 = 0;
	for (int i = 1; i < layers.size(); i++) {
		cout << i << endl;
		for (int j = 0; j < layers[i]->neurons.size(); j++) {
			for (int k = 0; k < layers[i]->neurons(j)->weight.size(); k++) {
				cout << abs(*layers[i]->neurons(j)->gradient(k) - theorGrads[i][j][k]) <<" "<< *layers[i]->neurons(j)->gradient(k) <<" "<<theorGrads[i][j][k]<< endl;
				avgDiff += pow((*layers[i]->neurons(j)->gradient(k)  - theorGrads[i][j][k]),2);
				++cnt;
				denom1 += pow(*layers[i]->neurons(j)->gradient(k), 2);
				denom2 += pow(theorGrads[i][j][k], 2);
				maxDiff = max(maxDiff, abs(*layers[i]->neurons(j)->gradient(k) - theorGrads[i][j][k]));
			}
		}
	}
	update(1);
	cout << maxDiff <<" maxDiff "<<sqrt(avgDiff)/(sqrt(denom1)+sqrt(denom2))<< endl;//if rval order 10-7 good
}
void Net::feedForward(Vector<double> &input, bool eval) {
	layers[0]->feedForward(input);
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->feedForward(layers[i-1]->neurons, dropOut,eval);
	}
}
void Net::printOut() {
	for (int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++) {
		cout << layers[layers.size() - 1]->neurons(i)->output << endl;
	}
}
void Net::printState() {
	for (int i = 1; i < layers.size(); i++) {
		cout << i;
		layers[i]->printState();

		
	}


}
void Net::printErr() {
	cout << layers[layers.size() - 1]->totalError() << endl;
}
double Net::getErr() {
	return layers[layers.size() - 1]->totalError();
}
double Net::eval(vector<Vector<double> > &inputs,vector<vector<double> > &targs, double threshold) {
	double cnt = 0;
	for (int i = 0; i < targs.size(); i++) {
		feedForward(inputs[i]);
		setTarget(targs[i]);
		if (layers[layers.size() - 1]->eval(threshold)) {
			++cnt;
		}
	}
	return cnt / targs.size();
}
void Net::printLayers() {
	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i]->neurons.size(); j++) {
			cout << layers[i]->neurons(j)->output << " ";
		}
		cout << "layer: "<<i << endl;
	}
}
void Net::saveNet(string fileName) {
	ofstream fout(fileName);
	fout << setprecision(9)<< showpoint <<layers.size()<<endl;
	for (int i = 0; i < layers.size(); i++) {
		fout << setprecision(9)<<layers[i]->type<<","<< layers[i]->neurons.size()<<","<< layers[i]->neurons(0)->weight.size() <<endl;
		if (layers[i]->type=="ConvLayer") {
			fout << layers[i]->neurons.z << endl;//numkernels
				fout << setprecision(9) << showpoint << layers[i]->neurons(0, 0, 0)->weight.x << "," << layers[i]->neurons(0, 0, 0)->weight.y;
			
			fout << endl;
			for (int c = 0; c < layers[i]->neurons.z; c++) {
				for (int j = 0; j < max(1, layers[i]->neurons(0,0,c)->weight.x); j++) {
					for (int k = 0; k < max(1, layers[i]->neurons(0,0,c)->weight.y); k++) {
						for (int h = 0; h < max(1, layers[i]->neurons(0,0,c)->weight.z); h++) {
							fout << setprecision(9) << showpoint << *layers[i]->neurons(0,0,c)->weight(j, k, h);//filterweights
							if (j != max(1, layers[i]->neurons(0,0,c)->weight.x) - 1 || k != max(1, layers[i]->neurons(0,0,c)->weight.y) - 1 || h != max(1, layers[i]->neurons(0,0,c)->weight.z) - 1)
								fout << ",";
						}
					}
				}
				fout << endl;
				fout << *layers[i]->neurons(0, 0, c)->bias;
				fout << endl;
			}
		}
		else if (layers[i]->type=="InLayer") {
			fout << setprecision(9) << showpoint<< layers[i]->neurons.size()<<endl;
		}
		else {
			for (int p = 0; p < layers[i]->neurons.size(); p++) {
				fout << setprecision(9) << layers[i]->neurons(p)->weight.size()<< "," << *layers[i]->neurons(p)->bias <<endl;
				for (int j = 0; j < layers[i]->neurons(p)->weight.size(); j++) {
					fout << setprecision(9)<<showpoint << *layers[i]->neurons(p)->weight(j);
					if (j != layers[i]->neurons(p)->weight.size() - 1)
						fout << ",";
				}
				fout << endl;
			}
		}
	}
}
Net::~Net()
{
	
}
