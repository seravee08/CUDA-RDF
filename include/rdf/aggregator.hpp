#ifndef AGGREGATOR_HPP
#define AGGREGATOR_HPP

#include <rdf/sample.hpp>
#include <vector>

namespace rdf {

    /**
     * TODO: class document...
     */
    class Aggregator {
    public:
        Aggregator(const int& binCount);
        Aggregator() {}
        ~Aggregator() {}        

        /**
         * TODO: function doc...
         */
        void         clear();        

        double       entropy() const;        

        /**
         * TODO: doc...
         * @param sample
         */
        void         initialize(const int& binCount);
        void         aggregate(const rdf::Sample& sample);
        void         aggregate(const rdf::Aggregator& aggregator);        

        Aggregator   clone() const;
        unsigned int sampleCount() const;
        int          binCount() const;
        unsigned int samplePerBin(int binIdx) const;
        void         setSampleCount(const unsigned int& sampleCount);
        void         setBin(int binIdx, const unsigned int& value);    
		void		 manualSet(unsigned int* bins, const int& binCount);

    private:
        std::vector<unsigned int>   bins_;
        int            binCount_;
        unsigned int   sampleCount_;
    };

    typedef boost::shared_ptr<Aggregator> AggregatorPtr;

} // namespace rdf

#endif
