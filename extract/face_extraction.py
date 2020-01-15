import os

import common_face_extraction as fe
from dnn_extraction import extractFacesDNN
from dnn_tracking_extraction import extractFacesDNNTracking



#INFO: using proposed method by sr6033
# https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
# for face extraction from video input

#TODO: could be issues with faces too close to the edges of the image
# in case of squared output. Might be a problem.

#Extraction method can be any of:
#   'DNN'            : using DNN
#   'DNN_TRACKING'   : using DNN with tracking

#Tracking type can be any of:
#   'MIL'
#   'BOOSTING'
#   'KCF'
#   'TLD'
#   'MEDIANFLOW'
#   'GOTURN'
#   'MOSSE'
#   'CSRT'

###DEFAULT CONFIGURATION###
defaultExtractionMethod =  'DNN'
defaultTrackerType = 'MIL'

modelDir = "caffemodel"
defaultPrototxt = modelDir+os.sep+"deploy.prototxt.txt"
defaultModel = modelDir+os.sep+"res10_300x300_ssd_iter_140000.caffemodel"

defaultNetSize = 300
defaultMean = (104.0, 177.0, 123.0)

defaultWidth = 300
defaultMinConfidence = 0.95

defaultFrameStep=1
defaultIsSquare = True
logEnabled = True


class FaceExtractor:
    @staticmethod
    def  extractFaces(
                    src,                                    #path to video source for extraction
                    extractionMethod =defaultExtractionMethod,#name of extraction method to be used
                    width            =defaultWidth,         #width of extracted face
                    isSquare         =defaultIsSquare,      #output face as a squared of dim width x width
                    frameStart       =0,                    #Frame at which to begin extraction
                    frameEnd         =None,                 #Frame at which to end
                    frameStep        =defaultFrameStep,     #read video every ... frames
                    maxFrames        =None,                 #maximum number of frames to be read

                    minConfidence    =defaultMinConfidence, #confidence threshold
                    prototxt         =defaultPrototxt,      #path to prototxt configuration file
                    model            =defaultModel,         #path to model
                    netSize          =defaultNetSize,       #size of the processing dnn
                    mean             =defaultMean,          #mean colour to be substracted
                    trackerType      =defaultTrackerType,   #WHEN TRACKING: tracker type such as MIL, Boosting...
                    isSaved          =defaultIsSquare,      #save image in output directory
                    outDir           =None,                  #output directory for faces
                    logEnabled       =logEnabled            #ouput log info
            ):
        extractionFunctor = ExtractionMethodFactory.GetFunctor(extractionMethod)
        if extractionMethod == 'DNN_TRACKING' :
            return extractionFunctor(
                src             =src,
                width           =width,
                isSquare        =isSquare,
                frameStart      =frameStart,
                frameEnd        =frameEnd,
                frameStep       =frameStep,
                maxFrames       =maxFrames,
                minConfidence   =minConfidence,
                prototxt        =prototxt,
                model           =model,
                netSize         =netSize,
                mean            =mean,
                trackerType     =trackerType,
                isSaved         =isSaved,
                outDir          =outDir,
                logEnabled      =logEnabled
                )
        else: #no tracking method
            return extractionFunctor(
                src             =src,
                width           =width,
                isSquare        =isSquare,
                frameStart      =frameStart,
                frameEnd        =frameEnd,
                frameStep       =frameStep,
                maxFrames       =maxFrames,
                minConfidence   =minConfidence,
                prototxt        =prototxt,
                model           =model,
                netSize         =netSize,
                mean            =mean,
                isSaved         =isSaved,
                outDir          =outDir,
                logEnabled      =logEnabled
                )


class ExtractionMethodFactory:
    @staticmethod
    def GetFunctor(extractionMethod):
        switch = {
            'DNN'           : extractFacesDNN,
            'DNN_TRACKING'  : extractFacesDNNTracking
        }
        return switch.get(extractionMethod, None)
