#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include <vector>

#include "facemark.hpp"

namespace bp = boost::python;
namespace np = bp::numpy;

#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
    static void init_ar()
{
#endif
    Py_Initialize();
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}


PyObject* FacemarkHandler::fit(PyObject* image,
                               cv::Rect face
)
{
    cv::Mat cppImage;
    std::vector<std::vector<cv::Point2f>> cppLandmarks;
    std::vector<cv::Rect> cppFace(1, face);
    cppImage = pbcvt::fromNDArrayToMat(image);
    bool ret = _facemark->fit(cppImage, cppFace, cppLandmarks);
    if( !ret && cppLandmarks.empty())
        return nullptr;

    cv::Mat cppMatLandmarks;

    cppMatLandmarks = buildMatFromVector(cppLandmarks.at(0));

    return pbcvt::fromMatToNDArray(cppMatLandmarks);
}


    BOOST_PYTHON_MODULE (landmark_detection) {
        init_ar();

        bp::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();


        bp::class_<FacemarkHandler>("LandmarkExtractor", bp::init<std::string>())
                .def("loadModel", &FacemarkHandler::loadModel)
                .def("fit", &FacemarkHandler::fit);
        bp::class_<cv::Rect>("CPPRect", bp::init<int, int, int, int>());


        //   class_<Point2f>("Point2f", init<int, int>());
        /*class_<Landmarks>("Landmarks", init<int>())
                .def("__len__", &Landmarks::size)
                .def("clear", &Landmarks::clear)
                .def("append", &StdItem<Landmarks>::add, with_custodian_and_ward<1,2>())
                .def("__getitem__", &StdItem<Landmarks>::get, return_value_policy<copy_non_const_reference>())
                .def("__setitem__", &StdItem<Landmarks>::set, with_custodian_and_ward<1,2>())
                .def("__delitem__", &StdItem<Landmarks>::del);
    */
    }
