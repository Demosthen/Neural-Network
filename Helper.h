#pragma once
using namespace std;
#define ppp pair<double,pair<double,pair<double,double> > >
class Helper
{
public:
	Helper();
	static bool inline checkBounds(shared_ptr<ppp> bounds, pair<double, double> pos){
		return bounds->first <= pos.first&&bounds->second.first <= pos.second&&bounds->second.second.first > pos.first&&bounds->second.second.second>pos.second;
	}
	~Helper();
};

