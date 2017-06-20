#include <rdf/image.hpp>

namespace rdf{
	void Image::setRadius(int radius){
	    radius_ = radius;
    }

    int Image::getRadius(){
	    return radius_;
    }


    void Image::setCenter(const cv::Vec2i& center){
	    center_ = center;
    }

    cv::Vec2i Image::getCenter(){
	    return center_;
    }

    void Image::setFileNames(const std::string& id, const std::string& ts){
    	id_ = id;
    	ts_ = ts;
    }

    void Image::getFileNames(std::string& id, std::string& ts){
    	id = id_;
    	ts = ts_;
    }
}