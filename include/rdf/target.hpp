#ifndef TARGET_HPP
#define TARGET_HPP

#include <vector>
#include <boost/shared_ptr.hpp>

namespace rdf {

	class Target {
	public:
		Target() {}
		Target(const int& num);
		~Target() {}
		void initialize(const int& num);
		std::vector<float>& Prob();
	private:
		std::vector<float> prob_;
		int num_;
	};

	typedef boost::shared_ptr<Target> TargetPtr;
} // namespace rdf


#endif