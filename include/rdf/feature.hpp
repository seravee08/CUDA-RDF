#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <rdf/sample.hpp>
#include <rdf/logSpace.hpp>

namespace rdf {

    class Feature {
    public:
        Feature(rdf::LogSpace& space);
        Feature() {
			x_  = -1.0;
			y_  = -1.0;
			xx_ = -1.0;
			yy_ = -1.0;
		}
        ~Feature() {}        

        float getX();
        float getY();
		float getXX();
		float getYY();
        void  setX(float x);
        void  setY(float y);
		void  setXX(float x);
		void  setYY(float y);
        void  copyTo(rdf::Feature& feature);
        float getResponse(const rdf::Sample& sample) const;        
		void  manualSet(float x, float y, float xx, float yy);

    private:
        //offsets
        float x_;
        float y_;
        //candidate offsets
        float xx_;
        float yy_;
    };

    typedef boost::shared_ptr<Feature> FeaturePtr;

} // namespace rdf


#endif // FEATURE_HPP
