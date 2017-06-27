#include <preprocess/math.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

#define off_position 3
#define pad 4


namespace preprocess {
	cv::Vec2i Math::calculateHandCenter(const RGBImage& rgb){
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		int count = 0;
		cv::Vec2i center(0, 0);
		for (int i = 0; i < rgb_.rows; ++i) {
			for (int j = 0; j < rgb_.cols; ++j) {
				if (rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240) {
					center[0] += i;
					center[1] += j;
					count++;
				}
			}
		}

		center[0] /= count;
		center[1] /= count;
		return center;
	}

	float Math::findHandMeanDep(const DepthImage& depth, const RGBImage& rgb) {
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		cv::Mat_<float> depth_ = depth.getDepth();
		std::vector<float> vals;
		for (int i = 0; i < rgb_.rows; i++) {
			for (int j = 0; j < rgb_.cols; j++) {
				if (rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240) {
					vals.push_back(depth_(i, j));
				}
			}
		}

		float sum = std::accumulate(vals.begin(), vals.end(), 0.0);
		float mean = sum / vals.size();
		return mean;
	}


	void Math::high_filter(DepthImage& depth) {
		const cv::Size kernelSize(9, 9);
		cv::Mat_<float> des;
		cv::Mat_<float> delta;
		cv::Mat_<float> depth_ = depth.getDepth();

		//cv::normalize(depth_, depth_, 0, 1, cv::NORM_MINMAX);

		cv::GaussianBlur(depth_, des, kernelSize, 0);

		if (depth_.rows != des.rows || depth_.cols != des.cols) {
			std::cout << "Demension should be the same!" << std::endl;
		}

		des = depth_ - des;

		cv::pow(des, 2, delta);
		cv::GaussianBlur(delta, delta, kernelSize, 0);
		cv::sqrt(delta, delta);
		float c = cv::mean(delta).val[0];
		delta = cv::max(delta, c);

		des = des / delta;
		//cv::divide(des, delta, des);
		depth.setDepth(des);
	}

	void Math::findHandMask(RGBImage& rgb) {
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();

		for (int i = 0; i < rgb_.rows; i++) {
			for (int j = 0; j < rgb_.cols; j++) {
				if (!(rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240)) {
					rgb_(i, j)[0] = 255;
					rgb_(i, j)[1] = 255;
					rgb_(i, j)[2] = 255;
				}
			}
		}

		rgb.setRGB(rgb_);
	}

	void Math::normalizeMinusOneToOne(DepthImage& depth) {
		depth.setDepth((depth.getDepth() + 1.0) / 2.0);
	}

	void Math::normalizeAll(DepthImage& depth) {
		cv::Mat_<float> depth_ = depth.getDepth();
		cv::normalize(depth_, depth_, 0, 1, cv::NORM_MINMAX);
		depth.setDepth(depth_);
	}

	void Math::normalizeHand(DepthImage& depth, const RGBImage& rgb) {
		cv::Mat_<float> depth_ = depth.getDepth();
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		float max = -10.0;
		float min = 10.0;

		for (int i = 0; i < depth_.rows; i++) {
			for (int j = 0; j < depth_.cols; j++) {
				if (rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240) {
					if (depth_(i, j) < min) {
						min = depth_(i, j);
					}
					if (depth_(i, j) > max) {
						max = depth_(i, j);
					}
				}
			}
		}

		for (int i = 0; i < depth_.rows; i++) {
			for (int j = 0; j < depth_.cols; j++) {
				if (rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240) {
					depth_(i, j) = (depth_(i, j) - min) / (max - min);
				}
				else {
					depth_(i, j) = 2.0;
				}
			}
		}

		depth.setDepth(depth_);
	}


	void Math::scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const float& ratio) {
		cv::Mat_<float> depth_;
		cv::resize(depth.getDepth(), depth_, cv::Size(), ratio, ratio, cv::INTER_AREA);
		for (int i = 0; i < depth_.rows; i++) {
			for (int j = 0; j < depth_.cols; j++) {
				depth_(i, j) /= ratio;
			}
		}

		std::vector<cv::Mat_<int> > rgbChannels;
		cv::split(rgb.getRGB(), rgbChannels);
		//we use float here, opencv has bugs when operating int type
		cv::Mat_<float> b = rgbChannels[0];
		cv::Mat_<float> g = rgbChannels[1];
		cv::Mat_<float> r = rgbChannels[2];

		cv::Mat_<float> r_, g_, b_;
		cv::resize(r, r_, cv::Size(), ratio, ratio, cv::INTER_AREA);
		cv::resize(g, g_, cv::Size(), ratio, ratio, cv::INTER_AREA);
		cv::resize(b, b_, cv::Size(), ratio, ratio, cv::INTER_AREA);

		std::vector<cv::Mat_<int> > rgbChannels_;
		cv::Mat_<cv::Vec3i> rgb_;
		rgbChannels_.push_back(b_);
		rgbChannels_.push_back(g_);
		rgbChannels_.push_back(r_);
		cv::merge(rgbChannels_, rgb_);


		depth.setDepth(depth_);
		rgb.setRGB(rgb_);

		std::vector<cv::Vec2f> anno_ = anno.getAnno();
		for (int i = 0; i < anno_.size(); i++) {
			anno_[i][0] *= ratio;
			anno_[i][1] *= ratio;
		}
		anno.setAnno(anno_);
	}


	void Math::scale(cv::Mat_<float>& depth, cv::Mat_<cv::Vec3i>& rgb, std::vector<cv::Vec2f>& anno, const float& ratio) {
		cv::resize(depth, depth, cv::Size(), ratio, ratio, cv::INTER_AREA);

		std::vector<cv::Mat_<int> > rgbChannels;
		cv::split(rgb, rgbChannels);
		//we use float here, opencv has bugs when operating int type
		cv::Mat_<float> b = rgbChannels[0];
		cv::Mat_<float> g = rgbChannels[1];
		cv::Mat_<float> r = rgbChannels[2];

		cv::Mat_<float> r_, g_, b_;
		cv::resize(r, r_, cv::Size(), ratio, ratio, cv::INTER_AREA);
		cv::resize(g, g_, cv::Size(), ratio, ratio, cv::INTER_AREA);
		cv::resize(b, b_, cv::Size(), ratio, ratio, cv::INTER_AREA);

		std::vector<cv::Mat_<int> > rgbChannels_;
		rgbChannels_.push_back(b_);
		rgbChannels_.push_back(g_);
		rgbChannels_.push_back(r_);
		cv::merge(rgbChannels_, rgb);

		for (int i = 0; i < anno.size(); i++) {
			anno[i][0] *= ratio;
			anno[i][1] *= ratio;
		}
	}


	void Math::scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& out_dimension) {
		const cv::Size outSize(out_dimension, out_dimension);
		float ratio = (float)out_dimension / (float)depth.getDepth().rows;
		cv::Mat_<float> depth_;
		cv::resize(depth.getDepth(), depth_, outSize, 0, 0, cv::INTER_AREA);

		std::vector<cv::Mat_<int> > rgbChannels;
		cv::split(rgb.getRGB(), rgbChannels);
		//we use float here, opencv has bugs when operating int type
		cv::Mat_<float> b = rgbChannels[0];
		cv::Mat_<float> g = rgbChannels[1];
		cv::Mat_<float> r = rgbChannels[2];

		cv::Mat_<float> r_, g_, b_;
		cv::resize(r, r_, outSize, 0, 0, cv::INTER_AREA);
		cv::resize(g, g_, outSize, 0, 0, cv::INTER_AREA);
		cv::resize(b, b_, outSize, 0, 0, cv::INTER_AREA);

		std::vector<cv::Mat_<int> > rgbChannels_;
		cv::Mat_<cv::Vec3i> rgb_;
		rgbChannels_.push_back(b_);
		rgbChannels_.push_back(g_);
		rgbChannels_.push_back(r_);
		cv::merge(rgbChannels_, rgb_);


		depth.setDepth(depth_);
		rgb.setRGB(rgb_);

		std::vector<cv::Vec2f> anno_ = anno.getAnno();
		for (int i = 0; i < anno_.size(); i++) {
			anno_[i][0] *= ratio;
			anno_[i][1] *= ratio;
		}
		anno.setAnno(anno_);
	}


	void Math::crop(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius) {
		const RGBImage const_rgb = rgb;
		cv::Vec2i center = Math::calculateHandCenter(const_rgb);
		cv::Vec2i offset(-radius, -radius);
		depth.setCenter(center);
		rgb.setCenter(center);

		cv::Vec2i origin = center + offset;
		int wid = radius * 2;

		//depth
		cv::Mat_<float> depth_ = depth.getDepth()(cv::Range(origin[0], origin[0] + wid), cv::Range(origin[1], origin[1] + wid));
		depth.setDepth(depth_);
		//rgb
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB()(cv::Range(origin[0], origin[0] + wid), cv::Range(origin[1], origin[1] + wid));
		rgb.setRGB(rgb_);
		//anno
		std::vector<cv::Vec2f> anno_ = anno.getAnno();
		for (unsigned int idx = 0; idx < anno_.size(); ++idx){
			anno_[idx][0] -= (float)origin[1];                //switch x and y
			anno_[idx][1] -= (float)origin[0];
		}
		anno.setAnno(anno_);
	}


	int Math::crop_test_1(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius) {
		const RGBImage const_rgb = rgb;
		cv::Vec2i center = Math::calculateHandCenter(const_rgb);
		cv::Vec2i offset(-radius, -radius);
		depth.setCenter(center);
		rgb.setCenter(center);

		cv::Vec2i origin = center + offset;
		int wid = radius * 2;

		std::string id;
		std::string ts;
		depth.getFileNames(id, ts);

		int height = depth.getDepth().rows;
		int width = depth.getDepth().cols;
		int rad = radius;
		int off1 = 0;
		int off2 = 0;

		if (origin[0] < 0 || origin[1] < 0) {
			off1 = std::min(origin[0], origin[1]);
			off1 = std::abs(off1);
		}

		if (origin[0] + wid > height || origin[1] + wid > width) {
			off2 = std::max(origin[0] + wid - height, origin[1] + wid - width);
		}

		return std::min(rad - off1, rad - off2);
	}

	int Math::crop_test_2(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius) {
		const RGBImage const_rgb = rgb;
		cv::Vec2i center = Math::calculateHandCenter(const_rgb);
		cv::Vec2i offset(-radius, -radius);
		depth.setCenter(center);
		rgb.setCenter(center);

		cv::Vec2i origin = center + offset;
		int wid = radius * 2;

		//anno
		std::vector<cv::Vec2f> anno_ = anno.getAnno();
		for (unsigned int idx = 0; idx < anno_.size(); ++idx){
			anno_[idx][0] -= (float)origin[1];                //switch x and y
			anno_[idx][1] -= (float)origin[0];
			if (anno_[idx][0] < 0.0 || anno_[idx][1] < 0.0 || anno_[idx][0] > (float)(wid - 1) || anno_[idx][1] >= (float)(wid - 1)) {
				return 1;
			}
		}

		return 0;
	}


	bool Math::pad_crop(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& radius) {
		const RGBImage const_rgb = rgb;
		cv::Vec2i center = Math::calculateHandCenter(const_rgb);
		cv::Vec2i offset(-radius, -radius);
		depth.setCenter(center);
		rgb.setCenter(center);

		cv::Mat_<float> depth_ = depth.getDepth();
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		std::vector<cv::Vec2f> anno_ = anno.getAnno();

		cv::Vec2i origin = center + offset;
		int wid = radius * 2;

		int height = depth.getDepth().rows;
		int width = depth.getDepth().cols;
		int rad = radius;
		int off1 = 0;
		int off2 = 0;

		for (unsigned int idx = 0; idx < anno_.size(); ++idx) {
			if (anno_[idx][0] < 0 || anno_[idx][1] < 0 || anno_[idx][0] > width - 1 || anno_[idx][1] > height - 1) {
				return false;
			}
		}

		if (origin[0] < 0 || origin[1] < 0) {
			off1 = std::min(origin[0], origin[1]);
			off1 = std::abs(off1);
		}

		if (origin[0] + wid > height || origin[1] + wid > width) {
			off2 = std::max(origin[0] + wid - height, origin[1] + wid - width);
		}

		int off = std::max(off1, off2);

		if (off > 0) {
			int top = off;
			int bottom = off;
			int left = off;
			int right = off;

			cv::copyMakeBorder(depth_, depth_, top, bottom, left, right, cv::BORDER_REPLICATE);
			cv::copyMakeBorder(rgb_, rgb_, top, bottom, left, right, cv::BORDER_REPLICATE);

			for (unsigned int idx = 0; idx < anno_.size(); ++idx){
				anno_[idx][0] += (float)off;
				anno_[idx][1] += (float)off;
			}

			origin[0] += (float)off;
			origin[1] += (float)off;
		}

		for (unsigned int idx = 0; idx < anno_.size(); ++idx){
			float ax = anno_[idx][0] - (float)origin[1];                //switch x and y
			float ay = anno_[idx][1] - (float)origin[0];

			if (ax < 0.0 || ay < 0.0 || ax >(float)(wid - 1) || ay >(float)(wid - 1)) {
				return false;
			}
		}

		//crop
		depth.setDepth(depth_(cv::Range(origin[0], origin[0] + wid), cv::Range(origin[1], origin[1] + wid)));
		rgb.setRGB(rgb_(cv::Range(origin[0], origin[0] + wid), cv::Range(origin[1], origin[1] + wid)));
		for (unsigned int idx = 0; idx < anno_.size(); ++idx){
			anno_[idx][0] -= (float)origin[1];                //switch x and y
			anno_[idx][1] -= (float)origin[0];
		}
		anno.setAnno(anno_);

		return true;
	}


	void Math::merge(const cv::Mat_<int>& r, cv::Mat_<int>& image) {
		int height = image.rows;
		int width = image.cols;

		// note here, source image and dest image have the same dimension
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (r(i, j) == 255)
					image(i, j) += r(i, j);
			}
		}
	}

	bool Math::isQualified(const RGBImage& rgb) {
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		int count = 0;
		for (int i = 0; i < rgb_.rows; ++i) {
			for (int j = 0; j < rgb_.cols; ++j) {
				if (rgb_(i, j)[0] < 20 && rgb_(i, j)[1] < 20 && rgb_(i, j)[2] > 240) {
					count++;
				}
			}
		}
		if (count < 2000)
			return false;
		else
			return true;
	}

	bool Math::GetSquareImageandAnno(cv::Mat_<float>& depth, cv::Mat_<cv::Vec3i>& rgb, std::vector<cv::Vec2f>& anno, const int& target_width) {
		int width = depth.cols, height = depth.rows;

		int top = 0, bottom = 0, left = 0, right = 0;

		for (int i = 0; i < anno.size(); i++) {
			if (anno[i][0] < 0 || anno[i][1] < 0) {
				return false;
			}
			if (anno[i][0] > depth.cols || anno[i][1] > depth.rows) {
				return false;
			}
		}

		if (width > height) {
			bottom = (width - height) / 2;
			top = width - height - bottom;
		}
		else if (width < height) {
			left = (height - width) / 2;
			right = height - width - left;
		}
		top += pad;
		bottom += pad;
		left += pad;
		right += pad;

		cv::Scalar depth_val(1.0);
		cv::Scalar rgb_val(255, 255, 255);
		//pad the image
		cv::copyMakeBorder(depth, depth, top, bottom, left, right, cv::BORDER_CONSTANT, depth_val);
		cv::copyMakeBorder(rgb, rgb, top, bottom, left, right, cv::BORDER_CONSTANT, rgb_val);

		for (int i = 0; i < anno.size(); i++) {
			anno[i][0] += left;
			anno[i][1] += top;
		}

		float ratio = (float)target_width / (float)depth.cols;

		scale(depth, rgb, anno, ratio);

		return true;
	}

	bool Math::findCandidates(DepthImage& depth, RGBImage& rgb, Anno& anno, const int& outSize) {
		cv::Mat handMask;
		cv::Mat_<float> depth_ = depth.getDepth();
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		std::vector<cv::Vec2f> anno_ = anno.getAnno();

		cv::inRange(rgb.getRGB(), std::vector<int> {0, 0, 240}, std::vector<int> {20, 20, 255}, handMask);

		std::vector<std::vector<cv::Point2i> > ctrs;
		int biggest = -1;
		int size = -1;
		cv::findContours(handMask, ctrs, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (int c = 0; c < ctrs.size(); c++) {
			if (cv::contourArea(ctrs[c]) > size) {
				biggest = c;
				size = cv::contourArea(ctrs[c]);
			}
		}
		cv::Rect box = cv::boundingRect(ctrs[biggest]);
		cv::Mat_<float> depth_roi = depth_(box);
		cv::Mat_<cv::Vec3i> rgb_roi = rgb_(box);
		for (int i = 0; i < anno_.size(); i++) {
			anno_[i][0] -= box.x;
			anno_[i][1] -= box.y;
		}

		bool valid = GetSquareImageandAnno(depth_roi, rgb_roi, anno_, outSize);
		if (valid) {
			depth.setDepth(depth_roi);
			rgb.setRGB(rgb_roi);
			anno.setAnno(anno_);
		}

		return valid;
	}


	void Math::offset(RGBImage& rgb) {
		cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
		cv::Mat_<cv::Vec3i> off = rgb_(cv::Range(0, rgb_.rows), cv::Range(1, rgb_.cols));
		cv::copyMakeBorder(off, off, 0, 0, 0, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
		rgb.setRGB(off);
	}
}