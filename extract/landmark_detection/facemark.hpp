#ifndef FIX_FACEMARK_HPP
#define FIX_FACEMARK_HPP

#pragma once
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "vec.hpp"

/* A face will be only its bounding box (openCV rectangle */
typedef cv::face::Facemark Network;

class FacemarkHandler
{
private:
    cv::Ptr<Network> _facemark;

public:
    FacemarkHandler();
    explicit FacemarkHandler(const std::string& model);
    void loadModel(const std::string& model) const;
    PyObject* fit(PyObject* image,
             cv::Rect face
             );

};

cv::Mat buildMatFromVector(const std::vector<cv::Point2f>& cppLandmarks);

#endif
