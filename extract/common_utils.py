
import cv2 #requires OpenCV version 3 !..

class Frame:
    # __image
    # __index
    # __to_detect : is the frame to be searched for faces ? !!TODO
    def __init__(self, image, index):
        self.__image = image
        self.__index = index

    @staticmethod
    def read(cap, index_frame):
        assert(cap.isOpened())
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame - 1)
        # reading next frame
        ok, image = cap.read()
        if not ok:
            # if no frame has been grabbed
            return False, None
        return True, Frame(image, index_frame)

    def image(self):
        return self.__image
    def index(self):
        return self.__index
    def w(self):
        return self.__image.shape[1]
    def h(self):
        return self.__image.shape[0]
    def dim(self):
        return tuple((self.w(), self.h()))
    def save(self, filepath):
        cv2.imwrite(filepath, self.image())


class Rectangle:
    #__x, __y, __w, __h

    def __init__(self, x, y, w, h):
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h

    def x(self):
        return self.__x
    def y(self):
        return self.__y
    def w(self):
        return self.__w
    def h(self):
        return self.__h
    def tuple(self):
        return (self.x1(), self.y1(), self.x2(), self.y2())

    def intersection(self, rect):
        if self.x() > rect.x():
            return rect.intersection(self)
        x_inter = max(0, self.x()+self.w() - rect.x())
        if self.y() <= rect.y():
            y_inter = max(0, self.y() + self.h() - rect.y())
            intersection = Rectangle(self.x(), self.y(), self.x() + x_inter, self.y() + y_inter)
        else:
            y_inter = max(0, rect.y() + rect.h() - self.y())
            intersection = Rectangle(self.x(), rect.y(), self.x() + x_inter, rect.y() + y_inter)
        return intersection

    def surface(self):
        return self.w()*self.h()

    def surface_intersection(self, rect):
        return self.intersection(rect).surface()

    @staticmethod
    def from_box(box):
        return Rectangle(box.x1(), box.y1(), box.x2() - box.x1(), box.y2() - box.y1())

class BoundingBox:
    #__x1, __y1, __x2, __y2

    def __init__(self, x1, y1, x2, y2):
        self.__x1 = x1
        self.__y1 = y1
        self.__x2 = x2
        self.__y2 = y2

    def x1(self):
        return self.__x1
    def y1(self):
        return self.__y1
    def x2(self):
        return self.__x2
    def y2(self):
        return self.__y2
    def w(self):
        return self.x2() - self.x1()
    def h(self):
        return self.y2() - self.y1()
    def tuple(self):
        return (self.x1(), self.y1(), self.x2(), self.y2())

    def enlarge(self, rate_enlarge):
        # enlarge box from centre
        centre = (self.x1() + self.w() // 2, self.y1() + self.h() // 2)
        x_enlarge = int((self.w() / 2) * (1 + rate_enlarge / 2))
        y_enlarge = int((self.h() / 2) * (1 + rate_enlarge / 2))
        self.__x1 = centre[0] - x_enlarge
        self.__y1 = centre[1] - y_enlarge
        self.__x2 = centre[0] + x_enlarge
        self.__y2 = centre[1] + y_enlarge

    def crop_image(self, image):
        # get bounding box dimensions
        (x1, y1, x2, y2) = self.tuple()
        return image[y1:y2, x1:x2]

    def square_image(self, image):
        (x1, y1, x2, y2) = self.tuple()
        w = x2 - x1
        h = y2 - y1
        # cropping as a square
        x_offset = max(0, (h - w) // 2)
        y_offset = max(0, (w - h) // 2)
        x1 = x1 - x_offset
        x2 = x2 + x_offset
        y1 = y1 - y_offset
        y2 = y2 + y_offset
        return BoundingBox(x1, y1, x2, y2).crop_image(image)


def read_frames_from_source(src,
                            start_frame,
                            end_frame,
                            step_frame,
                            max_frame):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError("Video could not be read at path: " + src)
    if end_frame is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = total_frames - 1
    list_frames = []
    index_frame = start_frame
    frame_count = 0
    k = 0
    #todo ::::!!!
    while (cap.isOpened()
           and index_frame < end_frame
           and (max_frame is None or frame_count < max_frame)):
        # getting right frame
        # with regards to start_frame, end_frame and step_frame
        index_frame = start_frame + step_frame * k
        ok, frame = Frame.read(cap, index_frame)
        if not ok:
            break;
        list_frames.append(frame)
        k+=1
    #freeing video
    cap.release()
    #returning list of frames
    return list_frames



def log(log_enabled, message):
    if log_enabled:
        print(message)
