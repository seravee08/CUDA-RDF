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
		static float findHandMeanDep(const DepthImage& depth, const RGBImage& rgb);
		static void high_filter(DepthImage& depth);
		static void normalizeHand(DepthImage& depth, const RGBImage& rgb);
		static void normalizeAll(DepthImage& depth);
		static void normalizeMinusOneToOne(DepthImage& depth);
		static void findHandMask(RGBImage& rgb);
		static void scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const float& ratio);
		static void scale(cv::Mat_<float>& depth, cv::Mat_<cv::Vec3i>& rgb, std::vector<cv::Vec2f>& anno, const float& ratio);
		static void scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& out_dimension);
		static void crop(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static int crop_test_1(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static int crop_test_2(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static bool pad_crop(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius);
		static void merge(const cv::Mat_<int>& r, cv::Mat_<int>& image);
		static void test_rotate(DepthImage& depth, RGBImage& rgb, Anno& anno);
		static cv::Point2f RotatePoint(const cv::Point2f& p, float rad);
		static cv::Point2f RotatePoint(const cv::Point2f& cen_pt, const cv::Point2f& p, float rad);
		static bool isQualified(const RGBImage& rgb);
		static bool GetSquareImageandAnno(cv::Mat_<float>& depth, cv::Mat_<cv::Vec3i>& rgb, std::vector<cv::Vec2f>& anno, const int& target_width);
		static bool findCandidates(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& outSize);
		static void offset(RGBImage& rgb);
	};

	typedef boost::shared_ptr<Math> MathPtr;
}

#endif