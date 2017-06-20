#include <rdf/logSpace.hpp>

#define base 10.0

namespace rdf {

	void LogSpace::initialize(const double& max, const int& spaceSize) {
		double max_ = std::log10(max);
		int halfSize = spaceSize / 2.0;
		double step = max_ / (double)(halfSize - 1);

		space_.resize(spaceSize);
		for(int i = 0; i < halfSize - 1; i++) {
			space_[i] = i * step;
		}
		space_[halfSize - 1] = max_;

		//compute log space
		for(int i = 0; i < halfSize; i++) {
			space_[i] = std::pow(base, space_[i]);
			space_[spaceSize - i - 1] = -space_[i];
		}

		std::sort(space_.begin(), space_.end());
	}

	std::vector<double>& LogSpace::getSpace() {
		return space_;
	}
}