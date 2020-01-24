import numpy as np
import cv2 #REQUIRES OpenCV 3

from . import common_utils as ut
from . import common_tracking as trck


class Face:
    # __box_original                   : Bounding box object of the face in original image
    # __frame_original                 : Frame object (w/ frame_index), diff dimensions than original
    # __landmarks             : array of features added before warping
    # __is_warped
    # __box_warped
    # __frame_warped
    def __init__(self, frame, box):
        self.__frame = frame
        self.__box = box
        self.__landmarks = None
        self.__is_warped = False
        self.__box_warped = None
        self.__frame_warped = None

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
    def frame(self):
        return self.__frame
    def index_frame(self):
        return self.__frame.index()
    def image(self):
        return self.__frame.image()
    def landmarks(self):
        return self.__landmarks

    def set_image(self, image):
        self.__frame = ut.Frame(image, self.index_frame(), to_search=True)
    def set_box(self, box):
        self.__box = box

    def set_warped(self, box_warped, image_warped):
        self.__set_box_warped(box_warped)
        self.__set_image_warped(image_warped)
        self.__is_warped = True

    def __set_image_warped(self, image_warped):
        self.__frame_warped = ut.Frame(image_warped, self.index_frame(), to_search=True)
    def __set_box_warped(self, box_warped):
        self.__box_warped = box_warped
    def set_features(self, landmarks):
        self.__landmarks = np.array(landmarks)

    def get_landmark_position(self, index_landmark):
        x, y = self.landmarks()[index_landmark][0], self.landmarks()[index_landmark][1]
        return ut.Point2D(x, y)
        #return self.features()[index_feature]

    def is_valid(self):
        if self.landmarks() is None:
            raise ValueError("Features need to be set before validation.")
        # TODO : FIND A BETTER CRITERIA
        return True

    def save(self, dir_out, are_landmarks_saved):
        if (not self.landmarks() is None) and are_landmarks_saved:
            self.__write_landmarks()
        self.__save_image(dir_out)

    def __save_image(self, dir_out):
        if self.__is_warped:
            self.__frame_warped.save(dir_out, self.__box_warped)
        else:
            self.frame().save(dir_out, self.box())

    def __write_landmarks(self, radius=2, colour=(0, 255, 0)):
        for i in range(len(self.landmarks())):
            pos = self.get_landmark_position(i)
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

    def save_faces(self, dir_out, are_saved_landmarks):
        for face in self.faces():
            face.save(dir_out, are_saved_landmarks)
