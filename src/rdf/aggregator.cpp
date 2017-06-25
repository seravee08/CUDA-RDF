#include <rdf/aggregator.hpp>

namespace rdf {
	Aggregator::Aggregator(const int& binCount) {
		binCount_ = binCount;
		bins_.resize(binCount_);
		for (int b = 0; b < binCount_; b++) {
			bins_[b] = 0;
		}
		sampleCount_ = 0;
	}

	void Aggregator::initialize(const int& binCount) {
		binCount_ = binCount;
		bins_.resize(binCount_);
		for (int b = 0; b < binCount_; b++) {
			bins_[b] = 0;
		}
		sampleCount_ = 0;
	}

	void Aggregator::clear(){
		for (int b = 0; b < binCount_; b++) {
			bins_[b] = 0;
		}
		sampleCount_ = 0;
	}

	double Aggregator::entropy() const {
		if (sampleCount_ == 0) {
			return 0.0;
		}

		double result = 0.0;
		for (int b = 0; b < binCount_; b++) {
			const double p = (double)bins_[b] / (double)sampleCount_;
			result -= (p == 0.0) ? 0.0 : p * log(p) / log(2.0);
		}
		return result;
	}

	void Aggregator::aggregate(const rdf::Sample& sample) {
		bins_[sample.getLabel()]++;
		sampleCount_++;
	}

	void Aggregator::aggregate(const rdf::Aggregator& aggregator){
		for (int b = 0; b < binCount_; b++) {
			bins_[b] += aggregator.bins_[b];
		}

		sampleCount_ += aggregator.sampleCount_;
	}

	void Aggregator::setBin(int binIdx, const unsigned int& value) {
		bins_[binIdx] = value;
	}

	Aggregator Aggregator::clone() const {
		rdf::Aggregator result(binCount_);
		for(int b = 0; b < binCount_; b++){
			result.setBin(b, bins_[b]);
		}

		result.setSampleCount(sampleCount_);

		return result;
	}

	unsigned int Aggregator::sampleCount() const {
		return sampleCount_;
	}

	int Aggregator::binCount() const {
		return binCount_;
	}

	unsigned int Aggregator::samplePerBin(int binIdx) const {
		return bins_[binIdx];
	}

	void Aggregator::setSampleCount(const unsigned int& sampleCount){
		sampleCount_ = sampleCount;
	}

	void Aggregator::manualSet(unsigned int* bins, const int& binCount) {
		int cntr = 0;
		binCount_ = binCount;
		bins_.resize(binCount);
		for (int i = 0; i < binCount; i++)
		{
			bins_[i] = bins[i];
			cntr += bins_[i];
		}
		sampleCount_ = cntr;
	}

} // namespace rdf
