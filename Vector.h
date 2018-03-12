
#include "stdafx.h"
#include <vector>
#pragma once
using namespace std;
template <class type>
class Vector
{
public:

	vector<type> data;
	int x = 0;
	int y = 0;
	int z = 0;
	int w = 0;
	Vector(){}
	inline Vector<type>(int X) {
		data.resize(X);
		x = X;
	}
	inline Vector<type>(int X, int Y) {
		data.resize(X*Y);
		x = X;
		y = Y;
	}
	inline Vector<type>(int X, int Y, int Z) {
		data.resize(X*Y*Z);
		x = X;
		y = Y;
		z = Z;
	}
	inline Vector<type>(int X, int Y, int Z, int W) {
		data.resize(X*Y*Z*W);
		x = X;
		y = Y;
		z = Z;
		w = W;
	}
	inline type & operator() (int a) {
		return data[a];
	}
	inline type & operator() (int a, int b) {
		return data[a*y + b];
	}
	inline type & operator() (int a, int b, int c) {
		return data[a*y*z + b*z + c];
	}
	inline type & operator() (int a, int b, int c, int d) {
		return data[a*y*z*w+b*z*w+c*w+d];
	}
	inline void push_back(type a) {
		data.push_back(a);
	}
	void inline assign(int a, type put) {
		data[a] = put;
	}
	void inline assign(int a, int b, type put) {
		data[a*y + b] = put;
	}
	void inline assign(int a, int b, int c, type put) {
		data[a*y*z + b*z + c] = put;
	}
	void inline assign(int a, int b, int c, int d, type put) {
		data[a*y*z*w + b*z*w + c*w + d] = put;
	}
	inline int getIndex(int a, int b, int c, int d) {
		return a*y*z*w + b*z*w + c*w + d;
	}
	inline int getIndex(int a, int b, int c) {
		return a*y*z + b*z + c;
	}
	inline void resize(int sz) {
		data.resize(sz);
		x = sz;
	}
	inline void resize(int sz, type val) {
		data.resize(sz, val);
		x = sz;
	}
	inline void resize(int sz1, int sz2, type val) {
		data.x = sz1;
		data.y = sz2;
		data.resize(sz1*sz2,val);

	}
	inline void reserve(int sz) {
		data.reserve(sz);
	}
	inline int size() {
		return data.size();
	}
	~Vector(){}
};
