#ifndef SAMPLE_HPP
#define SAMPLE_HPP

#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace rdf {

    /**
     * TODO: This is a sample...
     */
    class Sample {
    public:
        Sample() {}
        Sample(int x, int y, int idx);
        explicit Sample(cv::Point2i coor);        

        ~Sample() {}        

        void                setCoor(int x, int y);
        void                setCoor(const cv::Point2i& coor);
        cv::Point2i         getCoor() const;        

        void                setLabel(int label);
        int                 getLabel() const;        

        void                setIdx(int idx);
        int                 getIdx() const;        

        void                setDepth(const cv::Mat_<float>& depth);
        cv::Mat_<float>     getDepth() const;
        float               getDepth(int row, int col) const;        

		void				setDepthID(int id);
		int					getDepthID() const;

        void                setRGB(const cv::Mat_<cv::Vec3i>& rgb);
        cv::Mat_<cv::Vec3i> getRGB() const;
        cv::Vec3i           getRGB(int row, int col) const;        

    private:
        cv::Point2i         coor_;
        cv::Mat_<float>     depth_;
        cv::Mat_<cv::Vec3i> rgb_;
        int                 label_;
        int                 idx_;
		int					depth_id;
    };    

    typedef boost::shared_ptr<Sample> SamplePtr;

} // namespace rdf

#endif // SAMPLE_HPP
