#include <rdf/feature.hpp>

#include <opencv2/opencv.hpp>

#include <random>

namespace rdf {

	Feature::Feature(rdf::LogSpace& space) {
		std::vector<double>& space_ = space.getSpace();
		int num = space_.size();

		std::random_device rand;
		std::mt19937 gen(rand());
		std::uniform_int_distribution<> dis(0, num - 1);

		int idx_x = dis(gen);
		int idx_y = dis(gen);

		x_ = (float)space_[idx_x];
		y_ = (float)space_[idx_y];

		idx_x = dis(gen);
		idx_y = dis(gen);

		xx_ = (float)space_[idx_x];
		yy_ = (float)space_[idx_y];
	}

	float Feature::getX() {
		return x_;
	}

	float Feature::getY() {
		return y_;
	}

	float Feature::getXX() {
		return xx_;
	}

	float Feature::getYY() {
		return yy_;
	}

	void Feature::setX(float x) {
		x_ = x;
	}

	void Feature::setY(float y) {
		y_ = y;
	}

	void Feature::setXX(float x) {
		xx_ = x;
	}

	void Feature::setYY(float y) {
		yy_ = y;
	}

	void Feature::copyTo(rdf::Feature& feature) {
		feature.x_ = x_;
		feature.y_ = y_;
	}

	void Feature::manualSet(float x, float y, float xx, float yy) {
		x_ = x;
		y_ = y;
		xx_ = xx;
		yy_ = yy;
	}

    float Feature::getResponse(const rdf::Sample& sample) const {
        const cv::Point2i point = sample.getCoor();
        const float depth = sample.getDepth(point.y, point.x);    

        float x = x_ / depth + point.x;
        float y = y_ / depth + point.y;    

        x = (x < 0) ? 0 :
            (x >= sample.getDepth().cols) ? sample.getDepth().cols - 1 : x;    

        y = (y < 0) ? 0 :
            (y >= sample.getDepth().rows) ? sample.getDepth().rows - 1 : y;

        float xx = xx_ / depth + point.x;
        float yy = yy_ / depth + point.y;

        xx = (xx < 0) ? 0 :
            (xx >= sample.getDepth().cols) ? sample.getDepth().cols - 1 : xx;    

        yy = (yy < 0) ? 0 :
            (yy >= sample.getDepth().rows) ? sample.getDepth().rows - 1 : yy;


        std::random_device rand;
        std::mt19937 gen(rand());
        std::uniform_int_distribution<> dis(0, 1);

        int flag = dis(gen);

        if(flag == 1)
            return sample.getDepth(y, x) - depth;
        else
            return sample.getDepth(y, x) - sample.getDepth(yy, xx);
    }

} // namespace rdf
