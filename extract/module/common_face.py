import numpy as np
import cv2 #REQUIRES OpenCV 3

from . import common_utils as ut
from . import common_tracking as trck


class Face:
    # __box                   : Bounding box object of the face in original image
    # __frame                 : Frame object (w/ frame_index), diff dimensions than original
    # __features              : array of features added before warping
    def __init__(self, frame, box):
        self.__frame = frame
        self.__box = box
        self.__features = None

    def x1(self):
        return self.box().x1()
    def y1(self):
        return self.box().y1()
    def x2(self):
        return self.box().x2()
    def y2(self):
        return self.box().y2()
    def w(self):
        return self.box().w()
    def h(self):
        return self.box().h()

    def box(self):
        return self.__box
    def rectangle(self):
        return ut.Rectangle.from_box(self.box())
    def index_frame(self):
        return self.__frame.index()
    def image(self):
        return self.__frame.image()
    def features(self):
        return self.__features

    def set_image(self, image):
        self.__frame = ut.Frame(image, self.index_frame(), to_search=True)
    def set_features(self, features):
        self.__features = np.array(features)

    def get_feature_position(self, index_feature):
        x, y = self.features()[index_feature][0], self.features()[index_feature][1]
        return ut.Point2D(x, y)

    def is_valid(self):
        if self.features() is None:
            raise ValueError("Features need to be set before validation.")
        if len(self.features()) == 0:
            return False
        rel_box = ut.BoundingBox(0, 0, self.w(), self.h())
        for i in range(len(self.features())):
            pos = self.get_feature_position(i)
            if not pos.is_in(rel_box):
                # if feature was not found in the face (bounding box too small or face occlusion)
                return False
            i += 1
        return True

    def save_image(self, dir_out):
        self.__frame.save(dir_out, self.x1(), self.y1())

    def write_features(self, radius=2, colour=(0, 255, 0)):
        for i in range(len(self.features())):
            pos = self.get_feature_position(i)
            x, y = pos.tuple()
            cv2.circle(self.image(), (int(x), int(y)), radius, color=colour)
            i += 1

class Person:
    #__faces                 : list of Faces belonging to Person in video
    #__is_tracked            ; are they tracked?
    #__tracker               : tracker bound to Person
    #
    def __init__(self, face, frame, type_tracker, is_tracked=True):
        self.__faces = [face]
        #create and init tracker with first face's bounding box
        self.__is_tracked = is_tracked
        if self.__is_tracked:
            self.__tracker = trck.TrackerType.create_tracker(type_tracker)
            box = self.face(0).box()
            ok = self.__init_tracker(frame, box)
            if not ok:
                raise RuntimeError("Could not initialise tracker.")

    def __init_tracker(self, frame, box):
        return self.__tracker.init(frame.image(), box.tuple())

    def faces(self):
        return self.__faces
    def face(self, index):
        return self.__faces[index]
    def face_count(self):
        return len(self.__faces)

    def resized_width(self):
        #set by first face
        return self.face(0).resized_width()

    def __iter__(self):
        self.__it = 0
        return self
    def __next__(self):
        if self.__it >= self.face_count():
            raise StopIteration
        face = self.face(self.__it)
        self.__it += 1
        return face

    def append(self, face):
        self.__faces.append(face)
    def remove(self, face):
        self.__faces.remove(face)

    def set_face(self, index, face):
        self.__faces[index] = face


    def update_tracker(self, frame):
        #update bounding box from tracker
        ok, bad_box = self.__tracker.update(frame.image())
        if not ok:
            return False, None
        box = ut.BoundingBox(*bad_box)
        #failure of tracking as False
        return True, box

    def __update_faces(self, list_faces, box, frame):
        #LAST STEP: we get the face closest to tracker from list of faces
        rect = ut.Rectangle.from_box(box)
        face_closest = None
        max_surface = -1
        for face in list_faces:
            surface = face.rectangle().surface_intersection(rect)
            if (max_surface - 1) or (surface > max_surface):
                max_surface = surface
                face_closest = face
        if face_closest is None:
            return False
        self.append(face_closest)
        #Since the bounding box of the face changed,
        #we need to reset this person's tracker
        if self.__is_tracked:
            self.__init_tracker(frame, face_closest.box())
        return True

    def update(self, list_faces, frame):
        #need to update tracker first
        if self.__is_tracked:
            ok, box = self.update_tracker(frame)
            if not ok:
                return False
        ok = self.__update_faces(list_faces, box, frame)
        return ok

    def cull_faces(self):
        i = 0
        while i < self.face_count():
            face = self.face(i)
            if not face.is_valid():
                self.remove(face)
            else:
                i += 1

    def save_images(self, dir_out):
        for face in self.faces():
            #face.write_features()
            face.save_image(dir_out)
