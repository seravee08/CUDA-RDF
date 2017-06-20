#include <preprocess/math.hpp>
#include <algorithm>
#include <cmath>


namespace preprocess {
	cv::Vec2i Math::calculateHandCenter(const RGBImage& rgb){
	    cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
	    int count = 0;
	    cv::Vec2i center(0, 0);
	    for(int i = 0; i < rgb_.rows; ++i) {
		    for(int j = 0; j < rgb_.cols; ++j) {
			    if(rgb_(i, j)[0] == 0 && rgb_(i, j)[1] == 0 && rgb_(i, j)[2] == 255) {
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

    float Math::findHandMedianCoor(const DepthImage& depth, const RGBImage& rgb) {
    	cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
    	cv::Mat_<float> depth_ = depth.getDepth();
    	std::vector<float> vals;
    	for(int i = 0; i < rgb_.rows; i++) {
    		for(int j = 0; j < rgb_.cols; j++) {
    			if(rgb_(i, j)[0] == 0 && rgb_(i, j)[1] == 0 && rgb_(i, j)[2] == 255) {
    				vals.push_back(depth_(i, j));
    			}
    		}
    	}

    	std::sort(vals.begin(), vals.end());
    	int num = vals.size();
    	return vals[num / 2];
    }


    void Math::high_filter(DepthImage& depth) {
    	const cv::Size kernelSize(11, 11);
    	cv::Mat_<float> des;
    	cv::Mat_<float> delta;
    	cv::Mat_<float> depth_ = depth.getDepth();

    	//cv::normalize(depth_, depth_, 0, 1, cv::NORM_MINMAX);

    	cv::GaussianBlur(depth_, des, kernelSize, 0);

    	if(depth_.rows != des.rows || depth_.cols != des.cols ) {
    		std::cout << "Demension should be the same!" << std::endl;
    	}

    	des = depth_ - des;
    	 
    	cv::pow(des, 2, delta);
    	cv::GaussianBlur(delta, delta, kernelSize, 0);
    	cv::sqrt(delta, delta);
    	float c = cv::mean(delta).val[0];
    	delta = cv::max(delta, c);

    	des = des / delta;

    	depth.setDepth(des);
    }

    void Math::findHandMask(RGBImage& rgb) {
    	cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();

    	for(int i = 0; i < rgb_.rows; i++) {
    		for(int j = 0; j < rgb_.cols; j++) {
    			if(!(rgb_(i, j)[0] == 0 && rgb_(i, j)[1] == 0 && rgb_(i , j)[2] == 255)) {
    				rgb_(i, j)[0] = 255;
    				rgb_(i, j)[1] = 255;
    				rgb_(i, j)[2] = 255;
    			}
    		}
    	}

    	rgb.setRGB(rgb_);
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

    	for(int i = 0; i < depth_.rows; i++) {
    		for(int j = 0; j < depth_.cols; j++) {
    			if(rgb_(i, j)[0] == 0 && rgb_(i, j)[1] == 0 && rgb_(i , j)[2] == 255) {
    				if(depth_(i, j) < min) {
    					min = depth_(i, j);
    				}
    				if(depth_(i, j) > max) {
    					max = depth_(i, j);
    				}
    			}
    		}
    	}

    	for(int i = 0; i < depth_.rows; i++) {
    		for(int j = 0; j < depth_.cols; j++) {
    			if(rgb_(i, j)[0] == 0 && rgb_(i, j)[1] == 0 && rgb_(i , j)[2] == 255) {
    				depth_(i, j) = (depth_(i, j) - min) / (max - min);
    			}
    			else {
    				depth_(i, j) = 1.0;
    			}
    		}
    	}

    	depth.setDepth(depth_);
    }


    void Math::scale(DepthImage& depth, RGBImage& rgb, Anno& anno, const float& ratio) {
        cv::Mat_<float> depth_;
        cv::resize(depth.getDepth(), depth_, cv::Size(), ratio, ratio, cv::INTER_AREA);

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

        std::vector<cv::Vec3f> anno_ = anno.getAnno();
        for(int i = 0; i < anno_.size(); i++) {
            anno_[i][0] *= ratio;
            anno_[i][1] *= ratio;
        }
        anno.setAnno(anno_);
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

        std::vector<cv::Vec3f> anno_ = anno.getAnno();
        for(int i = 0; i < anno_.size(); i++) {
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
        std::vector<cv::Vec3f> anno_ = anno.getAnno();
        for(unsigned int idx = 0; idx < anno_.size(); ++idx){
            anno_[idx][0] -= origin[1];                //switch x and y
            anno_[idx][1] -= origin[0];
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

        if(origin[0] < 0 || origin[1] < 0) {
            off1 = std::min(origin[0], origin[1]);
            off1 = std::abs(off1);
        }

        if(origin[0] + wid > height || origin[1] + wid > width) {
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
        std::vector<cv::Vec3f> anno_ = anno.getAnno();
        int height = depth.getDepth().rows;
        int width = depth.getDepth().cols;
        for(unsigned int idx = 0; idx < anno_.size(); ++idx){
            anno_[idx][0] -= origin[1];                //switch x and y
            anno_[idx][1] -= origin[0];
            if(anno_[idx][0] < 0 || anno_[idx][1] < 0 || anno_[idx][0] >= wid || anno_[idx][1] >= wid) {
                return 1;
            }
        }
        
        return 0;
    }


    void Math::pad(DepthImage& depth, RGBImage& rgb, const int& radius) {
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

        if(origin[0] < 0 || origin[1] < 0) {
            off1 = std::min(origin[0], origin[1]);
            off1 = std::abs(off1);
        }

        if(origin[0] + wid > height || origin[1] + wid > width) {
            off2 = std::max(origin[0] + wid - height, origin[1] + wid - width);
        }

        int off = std::max(off1, off2);

        if(off > 0) {
            int top = off;
            int bottom = off;
            int left = off;
            int right = off;
    
            cv::Mat_<float> depth_ = depth.getDepth();
            cv::Mat_<cv::Vec3i> rgb_ = rgb.getRGB();
    
            cv::copyMakeBorder(depth_, depth_, top, bottom, left, right, cv::BORDER_REPLICATE);
            cv::copyMakeBorder(rgb_, rgb_, top, bottom, left, right, cv::BORDER_REPLICATE);

            depth.setDepth(depth_);
            rgb.setRGB(rgb_);
        }
    }
}