#ifndef LOGSPACE_HPP
#define LOGSPACE_HPP

#include <cmath>
#include <vector>
#include <iostream>

#include <boost/shared_ptr.hpp>

namespace rdf {

	class LogSpace {
	public:
		LogSpace() {}
		~LogSpace() {}
		void initialize(const double& max, const int& spaceSize);
		std::vector<double>& getSpace();
	private:
		std::vector<double> space_;
	};

	typedef boost::shared_ptr<LogSpace> LogSpacePtr;
} //namespace rdf

#endif