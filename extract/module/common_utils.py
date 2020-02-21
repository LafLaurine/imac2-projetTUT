import os
import numpy as np
import cv2  # requires OpenCV version 3

# using PNG lossless to save images
ext_codec_default = ".png"
param_codec_default = [cv2.IMWRITE_PNG_COMPRESSION, 0]


class Frame:
    # __image
    # __index
    # __to_search : is the frame to be searched for faces ?
    def __init__(self, image, index, to_search):
        self.__image = image
        self.__index = index
        self.__to_search = to_search

    @staticmethod
    def read_next(cap, index_frame, to_search):
        if not cap.isOpened():
            raise EOFError("Video capture is over.")
        # reading next frame
        ok, image = cap.read()
        if not ok:
            # if no frame has been grabbed
            return False, None
        return True, Frame(image, index_frame, to_search)

    def image(self):
        return self.__image

    def index(self):
        return self.__index

    def to_search(self):
        return self.__to_search

    def w(self):
        return self.__image.shape[1]

    def h(self):
        return self.__image.shape[0]

    def dim(self):
        return Point2D(self.w(), self.h())

    def get_cropped(self, box):
        x1, y1, x2, y2 = box.tuple()
        image_cropped = self.image()[y1:y2, x1:x2]
        return image_cropped

    def save(self, dir_out, coords, ext_codec=ext_codec_default, param_codec=param_codec_default):
        filename = "{}_x{}y{}".format(self.index(),coords.x(), coords.y())
        Frame.__save_image(self.image(), dir_out, filename, ext_codec, param_codec)

    def show(self, name_window):
        cv2.imshow(name_window, self.image())


    @staticmethod
    def __save_image(image, dir_out, filename, ext_codec, param_codec):
        # if output directory does not exist, create it
        create_dir(dir_out)
        filepath = os.path.join(dir_out, filename + ext_codec)
        # adding extension (OpenCV will encode accordingly)
        # saving output
        ok = cv2.imwrite(filepath, image, params=param_codec)
        if not ok:
            raise IOError("Could not save image at path: " + filepath)


class Rectangle:
    # __x, __y, __w, __h

    def __init__(self, x, y, w, h):
        self.__x = int(x)
        self.__y = int(y)
        self.__w = int(w)
        self.__h = int(h)

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def w(self):
        return self.__w

    def h(self):
        return self.__h

    def tuple(self):
        return self.x(), self.y(), self.w(), self.h()

    def intersection(self, rect):
        if self.x() > rect.x():
            return rect.intersection(self)
        x_inter = max(0, self.x() + self.w() - rect.x())
        if self.y() <= rect.y():
            y_inter = max(0, self.y() + self.h() - rect.y())
            intersection = Rectangle(self.x(), self.y(), self.x() + x_inter, self.y() + y_inter)
        else:
            y_inter = max(0, rect.y() + rect.h() - self.y())
            intersection = Rectangle(self.x(), rect.y(), self.x() + x_inter, rect.y() + y_inter)
        return intersection

    def surface(self):
        return self.w() * self.h()

    def surface_intersection(self, rect):
        return self.intersection(rect).surface()

    @staticmethod
    def from_box(box):
        return Rectangle(box.x1(), box.y1(), box.x2() - box.x1(), box.y2() - box.y1())


class BoundingBox:
    # __x1, __y1, __x2, __y2

    def __init__(self, x1, y1, x2, y2):
        self.__x1 = int(x1)
        self.__y1 = int(y1)
        self.__x2 = int(x2)
        self.__y2 = int(y2)

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
        return self.x1(), self.y1(), self.x2(), self.y2()
    def list(self):
        return [self.x1(), self.y1(), self.x2(), self.y2()]

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

    @staticmethod
    def from_rectangle(rect):
        return BoundingBox(rect.x(), rect.y(), rect.x() + rect.w(), rect.y() + rect.h())


class Point2D:
    # __x
    # __y
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def __add__(self, point):
        return Point2D(self.x() + point.x(), self.y() + point.y())

    def __sub__(self, point):
        return Point2D(self.x() - point.x(), self.y() - point.y())

    def __mul__(self, alpha):
        return Point2D(self.x() * alpha, self.y() * alpha)

    def element_wise_prod(self, point):
        return Point2D(self.x() * point.x(), self.y() * point.y())

    def tuple(self):
        return self.x(), self.y()

    def list(self):
        return [self.x(), self.y()]

    def is_in(self, box: BoundingBox):
        return (box.x1() <= self.x() <= box.x2()
                and box.y1() <= self.y() <= box.y2())

    @staticmethod
    def average(list_points):
        res = Point2D(0, 0)
        for point in list_points:
            res += point
        return res * (1 / len(list_points))

    @staticmethod
    def build_from_array(array_coords):
        list_points = []
        for i in range(len(array_coords)):
            x, y = array_coords[i][0], array_coords[i][1]
            list_points.append(Point2D(x, y))
        return list_points


def read_frames_from_source(src,
                            start_frame,
                            end_frame,
                            step_frame,
                            max_frame,
                            to_track):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError("Video could not be read at path: " + src)
    # checking step_frame
    if step_frame < 0:
        raise ValueError("Step should be strictly positive integer.")
    # TODO: Actually catching these exceptions
    # setting end_frame
    if end_frame is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = total_frames - 1
    # jumping to start_frame
    jump_forward(cap, start_frame)
    if to_track:
        # TODO: CHECK WITH MULTIPLE PEOPLE
        # from_capture_tracking is just too long to compute
        functor_read_frames = read_frames_from_capture_tracking
    else:
        functor_read_frames = read_frames_from_capture_base
    list_frames = functor_read_frames(cap, start_frame, end_frame, step_frame, max_frame)
    # freeing video
    cap.release()
    return list_frames

def jump_forward(cap, jump_frame):
    # we jump to the frame that's step_frame ahead
    ok = True
    i = 0
    while (i < jump_frame - 1) and ok:
        ok, placeholder = cap.read()
        i += 1
    return ok


def read_frames_from_capture_base(cap,
                                  start_frame,
                                  end_frame,
                                  step_frame,
                                  max_frame):
    list_frames = []
    index_frame = start_frame
    frame_count = 0
    while (cap.isOpened()
           and index_frame < end_frame
           and (max_frame is None or frame_count < max_frame)):
        # jumpin' to the next frame to be read -> skipping step_frame - 1 frames
        jump_forward(cap, step_frame)
        # computin' index
        index_frame = start_frame + step_frame * frame_count
        ok, frame = Frame.read_next(cap, index_frame, to_search=True)
        if not ok:
            break
        list_frames.append(frame)
        frame_count += 1
    # returning list of frames
    return list_frames


def read_frames_from_capture_tracking(cap,
                                      start_frame,
                                      end_frame,
                                      step_frame,
                                      max_frame):
    list_frames = []
    index_frame = start_frame
    frame_count = 0
    while (cap.isOpened()
           and index_frame < end_frame
           and (max_frame is None or frame_count < max_frame)):
        # we read every frame from start to finish
        # but only if frame_index = start_frame + step_frame * k
        # do we tag it as one to be searched
        to_search = ((index_frame - start_frame) % step_frame == 0)
        ok, frame = Frame.read_next(cap, index_frame, to_search)
        if not ok:
            break
        list_frames.append(frame)
        index_frame += 1
        frame_count += 1
    return list_frames


def create_dir(dir_path):
    # if output directory does not exist, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not os.path.isdir(dir_path):
        raise NotADirectoryError(dir_path)

def get_path_without_basedir(path_in):
    path_out = ""
    it_path = path_in
    while it_path:
        head, tail = os.path.split(it_path)
        if not head:
            break
        path_out = os.path.join(tail, path_out)
        it_path = head
    return path_out


def log(log_enabled, message):
    if log_enabled:
        print(message)
