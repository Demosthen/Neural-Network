#include "stdafx.h"
#include "Net.h"

using namespace std;
int x1 = 80;
int y2 = 60;
inline string trim(string str) {
	int i = 0;
	for (; i<str.size(); i++) {
		if (!isspace(str[i])) {
			break;
		}

	}
	string trimmed = str.substr(i);//i
	i = str.size();
	for (; i<str.size(); i--) {
		if (isspace(str[i])) {
			break;
		}

	}
	trimmed = trimmed.substr(0, i+1);
	return trimmed;
}
pair<int, int> slideWindow(Net& n , Vector<double> &img, pair<int,int> imgSize, int stride=40) {
	int temp = 0;
	double maxim = 0;
	int cnt = 0;
	pair<int, int> loc = { 0,0 };
	for (int i = 0; i < img.x-imgSize.first+stride/2; i+=stride) {
		for (int j = 0; j < img.y-imgSize.second+stride/2; j+=stride) {
			*n.layers[0]->bounds={ (double)i,{(double)j,{(double)min(imgSize.first + i,img.x),(double)min(imgSize.second + j,img.y)}} };
			n.feedForward(img,true);
			double p = n.layers[n.layers.size() - 1]->neurons(1)->output - (n.layers[n.layers.size() - 1]->neurons(0)->output);//+ n.layers[n.layers.size() - 1]->neurons(1)->output);
			
			cout << "p"<<i<<" "<<j<<" "<< (double)min(imgSize.first + i, img.x) <<" "<< (double)min(imgSize.second + j, img.y) <<" "<<p<< endl;
			if (p>maxim) {
				maxim = p;
				cnt = 1;
				loc.first = (i + min(imgSize.first + i, img.x)) / 2;
				loc.second = (j + min(imgSize.second + j, img.y)) / 2;
			}
			/*if (p >0.9) {
				++cnt;
				loc.first = (loc.first + (i + min(imgSize.first + i, img.x)) / 2);
				loc.second = (loc.second + (j + min(imgSize.second + j, img.y)) / 2);
			}*/
		}
	}
	loc.first /= max(cnt,1);
	loc.second /= max(cnt,1);
	return loc;
}
void doMnist(Net &n){
	ifstream train("mnist_train.csv");
	ifstream test("mnist_test.csv");
	vector<Vector<double> > trainSet;
	trainSet.resize(60000);
	vector<Vector<double> > testSet;
	testSet.resize(10000);
	vector<double> trainLabels;
	trainLabels.resize(60000);
	vector<double > targ;
	vector<double> t(10, 0);
	targ.resize(10, 0);
	vector<vector<double> > testLabels;
	testLabels.resize(10000,t);
	string line;
	int i = 0;
	while (getline(train, line)) {
		trainSet[i].resize(28 * 28);
		trainSet[i].x = 28;
		trainSet[i].y = 28;
		trainSet[i].z = 1;
		trainSet[i].w = 1;
		stringstream ss;
		ss << line;
		getline(ss, line, ',');
		trainLabels[i]=atoi(line.c_str());
		int j = 0;
		while(getline(ss,line,',')){
			trainSet[i]((j)/28, (j)%28)=((double)atoi(line.c_str()))/255;
			++j;
		}
		++i;
	}
	i = 0;
	while (getline(test, line)) {
		testSet[i].resize(28 * 28);
		testSet[i].x = 28;
		testSet[i].y = 28;
		testSet[i].z = 1;
		testSet[i].w = 1;
		stringstream ss;
		ss << line;
		getline(ss, line, ',');
		testLabels[i][atoi(line.c_str())]=1;
		int j = 0;
		while (getline(ss, line, ',')) {
			testSet[i]((j) / 28, (j) % 28) = ((double)atoi(line.c_str()))/255;
			++j;
		}
		++i;
	}
	int cnt = 0;
	int numEpochs = 20;
	for (int j = 0; j < numEpochs; j++) {
		for (int i = 0; i < trainSet.size(); i++) {
			int img = rand() % 60000;
			n.feedForward(trainSet[img]);
			++cnt;
			targ[trainLabels[img]] = 1;
			//n.testGrad(trainSet[i], targ);
			n.backProp(targ);
			n.update(cnt);
			targ[trainLabels[img]] = 0;

			if (i % 5000 == 4999) {
				n.printErr();
				cout << n.eval(testSet, testLabels, 0.5) << endl;
			}
		}
	}
}
void doSimpleTest(Net &n) {
	Vector<double> data;
	data.data = { 1,0,1,2,0,2,3,1,3,4,5,7,8,9,1,7,4,5,1,7,3,8,3,7,1,0,4 };
	data.x = 3;
	data.y = 3;
	data.z = 3;
	vector<double> targ = { 0,0,0,1 };
	for (int i = 0; i < 10; i++) {
		n.feedForward(data);
		n.backProp(targ);
		n.update(i + 1);
		n.printErr();
	}
	
}
void doSimpleTestGrad() {
	Vector<double> x;
	x.data = { 4,9,6,7,1,3,3,2,0,26,8,37,12,14,9,1,2,15,3,29,16,16,100,0,1,8,46,6,74,54,27,2 };

	x.x = 4;
	x.y = 4;
	x.z = 2;
	x.w = 1;
	pair<int, pair<int, int> > dims = { 4,{ 4,2 } };
	vector<shared_ptr<Layer> > layers;
	layers.resize(3);
	int arr[] = { 10 };
	//layers[0] = shared_ptr<Layer>(new ConvLayer(make_pair(dims.first, dims.second.first), 2, 1, 2, false, arr[0],dims.second.second));
	//layers[0] = shared_ptr<Layer>(new InLayer(4*4*2));
	layers[0] = shared_ptr<Layer>(new ConvLayer(make_pair(dims.first, dims.second.first), 2, 1, 2, false, arr[0], dims.second.second));

	layers[1] = shared_ptr<Layer>(new Layer(layers[0]->neurons.size(), 3, 2));
	//layers[1] = shared_ptr<Layer>(new MaxPool(2, layers[0]->neurons.x, layers[0]->neurons.y, layers[0]->neurons.z));
	//layers[1] = shared_ptr<Layer>(new ConvLayer(make_pair(dims.first, dims.second.first), 2, 1, 2, false, arr[0],dims.second.second));

	layers[2] = shared_ptr<Layer>(new OutLayer(layers[1]->neurons.size(), 3, 2));
	vector<double> targ = { 0,1,0 };
	Net network(0.001, 1000000, 0.9, 0.999, 0, layers);
	network.testGrad(x, targ);
	cin.get();
	network.testGrad(x, targ);
}

void doCifar10() {
	unordered_map<string, int> labelMap;
	labelMap["airplane"] = 0;
	labelMap["automobile"] = 1;
	labelMap["bird"] = 2;
	labelMap["cat"] = 3;
	labelMap["deer"] = 4;
	labelMap["dog"] = 5;
	labelMap["frog"] = 6;
	labelMap["horse"] = 7;
	labelMap["ship"] = 8;
	labelMap["truck"] = 9;
	unordered_map<int, bool> testMap;
	vector<Vector<double> > testSet;
	vector<vector<double> > testLabs;
	int numTest = 10000;
	int numEpochs = 10;
	testSet.resize(numTest);
	testLabs.resize(numTest);
	vector<int> labels;
	vector<double> label;//placeholder target
	label.resize(10, 0);
	labels.resize(60000, 0);
	int numIm = 60000;
	ifstream labs("C:\\Users\\Austin\\source\\repos\\imageConverter\\imageConverter\\labs.txt");
	string lin;
	cout << "labs" << endl;
	while (getline(labs, lin)) {
		stringstream ss;
		ss << lin;
		while (getline(ss, lin, ',')) {
			lin = trim(lin);
			int num = strtod(lin.substr(0, lin.find('_')).c_str(), nullptr);
			string lab = lin.substr(lin.find('_') + 1);
			labels[num] = labelMap[lab];
		}
	}
	string li;
	pair<int, pair<int, int> > dims = { 32,{ 32,3 } };
	cout << "getTest" << endl;
	for (int i = 0; i < numTest; i++) {
		int im = rand() % numIm;
		if (testMap[im]) {
			--i;
			continue;
		}
		label[labels[im]] = 1;
		testLabs[i] = label;
		label[labels[im]] = 0;
		testSet[i].resize(dims.first*dims.second.first*dims.second.second);
		testSet[i].x = dims.first;
		testSet[i].y = dims.second.first;
		testSet[i].z = dims.second.second;
		testMap[im] = true;
		string file = "C:\\Users\\Austin\\source\\repos\\imageConverter\\imageConverter\\trainImgs\\" + to_string(im);
		file += "_img.txt";
		ifstream fin(file);
		int x = 0;
		while (getline(fin, li)) {
			stringstream ss;
			ss << li;
			int y = 0;
			while (getline(ss, li, ',')) {
				testSet[i](x, y / 3, y % 3) = strtod(li.c_str(), nullptr);
				++y;
			}
			++x;
		}

	}
	vector<shared_ptr<Layer> > layers;
	layers.resize(5);
	int arr[] = { 10,-1, 20,20,10 };
	int lay = 0;
	//cout << "layers" << endl;
	//layers[0] = shared_ptr<InLayer>(new InLayer(32 * 32 * 3));
	layers[lay] = shared_ptr<Layer>(new ConvLayer(make_pair(dims.first, dims.second.first), 3, 2, 2, false, arr[lay],dims.second.second));
	++lay;
	layers[lay] = shared_ptr<Layer>(new MaxPool(2, dims.first, dims.first, dims.second.second));
	++lay;
	layers[lay] = shared_ptr<Layer>(new Layer(layers[lay-1]->neurons.size(), arr[lay], 2));
	++lay;
	//cout << "layer" << endl;
	layers[lay] = shared_ptr<Layer>(new Layer(layers[lay-1]->neurons.size(), arr[lay], 2));
	++lay;
	//layers[2] = shared_ptr<Layer>(new Layer(layers[1]->neurons.size(), arr[2], 2));
	//cout << "outlayer" << endl;
	layers[lay] = shared_ptr<Layer>(new OutLayer(layers[lay-1]->neurons.size(), arr[lay], 2));
	//cout << "net" << endl;
	Net network(0.001, 1000000, 0.9, 0.999, 0, layers);//0.001
	srand(time(NULL));
	int total = 0;
	cout << "loop" << endl;
	for (int n = 0; n < numEpochs; n++) {
		for (int i = 0; i < numIm; i++) {

			int im = rand() % numIm;
			if (testMap[im]) {
				i--;
				continue;
			}
			++total;
			string file = "imageConverter\\imageConverter\\trainImgs\\" + to_string(im);
			file += "_img.txt";
			ifstream fin(file);
			Vector<double> input(dims.first, dims.second.first, dims.second.second);
			input.x = dims.first;
			input.y = dims.second.first;
			input.z = dims.second.second;
			input.w = 1;
			string line;
			int x = 0;
			//cout << "hey" << endl;
			while (getline(fin, line)) {

				stringstream ss;
				ss << line;
				int y = 0;
				while (ss, line, ',') {
					input(x, y / 3, y % 3) = strtod(line.c_str(), nullptr);
					++y;
				}
				++x;
			}
			//cout << "feed" << endl;
			network.feedForward(input);
			label[labels[im]] = 1;
			//cout << labels[im] << endl;
			//cout << "back" << endl;
			//network.testGrad(input, label);
			network.backProp(label);
			label[labels[im]] = 0;
			network.update(total);
			//cout << "err" << endl;
			if (i % 5000 == 4999) {
				network.printErr();
				//cout<<network.eval(testSet, testLabs, 100000)<<endl;
			}
		}
	}
}
int main() {

	srand(time(NULL));
	//doSimpleTestGrad();
	//cin.get();
	vector<shared_ptr<Layer> > layers;
	layers.resize(4);
	int arr[] = { 5,15,30,2 };
	int lay = 0;
	layers[lay] = shared_ptr<Layer>(new ConvLayer(make_pair(x1, y2), 3, 2, 2, false, arr[lay],3) );
	++lay;//layers[0] = shared_ptr<Layer>(new InLayer(28 * 28));
	//cout << "layer" << endl;
	layers[lay] = shared_ptr<Layer>(new Layer(layers[lay-1]->neurons.size(), arr[lay], 2));
	//layers[2] = shared_ptr<Layer>(new Layer(layers[1]->neurons.size(), arr[2], 2));
	//cout << "outlayer" << endl;
	//layers[2] = shared_ptr<Layer>(new OutLayer(layers[1]->neurons.size(), arr[2], 2));
	++lay;
	layers[lay] = shared_ptr<Layer>(new Layer(layers[lay - 1]->neurons.size(), arr[lay], 2));

	++lay;
	layers[lay] = shared_ptr<Layer>(new OutLayer(layers[lay-1]->neurons.size(), arr[lay],1));
	//cout << "net" << endl;
	
	Net network(0.001, 1000000, 0.9, 0.999, 0.2, layers);//0.001
	//doSimpleTestGrad();
	//doSimpleTest(network);
	//doMnist(network);
	//doCifar10();
	network.saveNet("network.txt");
	cin.get();
	//TODO:softmax, cross entropy loss
	return 0;
}