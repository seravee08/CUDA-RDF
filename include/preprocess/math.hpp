#ifndef MATH_HPP
#define MATH_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <preprocess/depthImage.hpp>
#include <preprocess/rgbImage.hpp>
#include <preprocess/anno.hpp>
#include <boost/shared_ptr.hpp>

namespace preprocess {
	class Math {
	public:
		Math() {};
		~Math() {};
		static cv::Vec2i calculateHandCenter(const RGBImage& rgb);
		static float findHandMedianCoor(const DepthImage& depth, const RGBImage& rgb);
		static void high_filter(DepthImage& depth);
		static void normalizeHand(DepthImage& depth, const RGBImage& rgb);
		static void normalizeAll(DepthImage& depth);
		static void findHandMask(RGBImage& rgb);
		static void scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const float& ratio);
		static void scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& out_dimension);
		static void crop(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static int crop_test_1(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static int crop_test_2(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static void pad(DepthImage& depth, RGBImage& rgb, const int& radius);
	};

	typedef boost::shared_ptr<Math> MathPtr;
}

#endif