#include <iostream>


#include "facemark.hpp"


FacemarkHandler::FacemarkHandler() : _facemark(cv::face::createFacemarkLBF()) {}

FacemarkHandler::FacemarkHandler(const std::string& model) : FacemarkHandler()
{
    loadModel(model);
}

void FacemarkHandler::loadModel(const std::string& model) const
{
    _facemark->loadModel(model);
}

cv::Mat buildMatFromVector(const std::vector<cv::Point2f>& cppLandmarks)
{
    auto cppMatLandmarks = cv::Mat(cppLandmarks.size(), 2, CV_64F);
    int i=0;
    for(const auto& landmark : cppLandmarks)
    {
        cppMatLandmarks.at<double>(i,0) = landmark.x;
        cppMatLandmarks.at<double>(i,1) = landmark.y;
        ++i;
    }
    return cppMatLandmarks;
}