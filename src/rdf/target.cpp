#include <rdf/target.hpp>


namespace rdf {

	Target::Target(const int& num) {
		num_ = num;
		prob_.resize(num_);
		for(int i = 0; i < num_; i ++) {
			prob_[i] = 0;
		}
	}

	std::vector<float>& Target::Prob() {
		return prob_;
	}

	void Target::initialize(const int& num) {
		num_ = num;
		prob_.resize(num_);
		for(int i = 0; i < num_; i ++) {
			prob_[i] = 0;
		}
	}
}