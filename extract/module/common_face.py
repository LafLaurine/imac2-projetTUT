import numpy as np
import cv2 #REQUIRES OpenCV 3

from . import common_utils as ut
from . import common_tracking as trck


class Face:
    # __box_original                   : Bounding box object of the face in original image
    # __frame_original                 : Frame object (w/ frame_index), diff dimensions than original
    # __landmarks_original             : array of features added before warping
    # __is_warped
    # __box_warped
    # __frame_warped
    # __landmarks_warped           : array of features added before warping
    __RADIUS_LANDMARK = 2
    __THICKNESS_LANDMARK = 3
    __COLOUR_LANDMARK = (0, 255, 0)
    __THICKNESS_RECTANGLE = 6
    __COLOUR_RECTANGLE = (127, 0, 255)
    def __init__(self, frame, box):
        self.__frame_original = frame
        self.__box_original = box
        self.__landmarks_original = None
        self.__is_warped = False
        self.__box_warped = None
        self.__frame_warped = None
        self.__landmarks_warped = None

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
        box = self.__box_original if not self.__is_warped else self.__box_warped
        return box
    def rectangle(self):
        return ut.Rectangle.from_box(self.box())
    def frame(self):
        frame = self.__frame_original if not self.__is_warped else self.__frame_warped
        return frame
    def index_frame(self):
        return self.frame().index()
    def image(self):
        return self.frame().image()
    def landmarks(self):
        landmarks = self.__landmarks_original if not self.__is_warped else self.__landmarks_warped
        return landmarks

    def set_image(self, image):
        self.__frame = ut.Frame(image, self.index_frame(), to_search=True)
    def set_box(self, box):
        self.__box = box

    def set_warped(self, box_warped, image_warped, landmarks_warped):
        self.__set_box_warped(box_warped)
        self.__set_image_warped(image_warped)
        self.__set_landmarks_warped(landmarks_warped)
        self.__is_warped = True

    def __set_image_warped(self, image_warped):
        self.__frame_warped = ut.Frame(image_warped, self.index_frame(), to_search=True)
    def __set_box_warped(self, box_warped):
        self.__box_warped = box_warped
    def set_landmarks_original(self, landmarks_original):
        self.__landmarks_original = np.array(landmarks_original)

    def __set_landmarks_warped(self, landmarks_warped):
        self.__landmarks_warped = np.array(landmarks_warped)

    def is_valid(self):
        if self.landmarks() is None:
            raise ValueError("Features need to be set before validation.")
        # TODO : FIND A BETTER CRITERIA
        return True

    def save(self, dir_out, are_landmarks_saved=False, is_rectangle_saved=False):
        self.__write_to_image(self.image(), are_landmarks_saved, is_rectangle_saved)
        self.__save_frame(dir_out)

    def write_to_frame(self, frame, are_landmarks_saved=False, is_rectangle_saved=False):
        self.__write_to_image(frame.image(), are_landmarks_saved, is_rectangle_saved)


    def __write_to_image(self, image, are_landmarks_saved, is_rectangle_saved):
        if (self.landmarks() is not None) and are_landmarks_saved:
            self.__write_landmarks_to_image(image)
        if is_rectangle_saved:
            self.__write_box_to_image(image)

    def __save_frame(self, dir_out):
        coords = ut.Point2D(self.__box_original.x1(), self.__box_original.y1())
        self.frame().save(dir_out, coords)

    def __write_landmarks(self):
       self.__write_landmarks_to_image(self.image())

    def __write_box(self):
        self.__write_box_to_image(self.image())

    def __write_landmarks_to_image(self, image):
        landmarks = self.landmarks()
        for x, y in landmarks:
            cv2.circle(image, (int(x), int(y)), radius=Face.__RADIUS_LANDMARK, thickness=Face.__THICKNESS_LANDMARK, color=Face.__COLOUR_LANDMARK)

    def __write_box_to_image(self, image):
        pt1 = (self.x1(), self.y1())
        pt2 = (self.x2(), self.y2())
        cv2.rectangle(image, pt1, pt2, thickness=Face.__THICKNESS_RECTANGLE, color=Face.__COLOUR_RECTANGLE)





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
            if (max_surface == - 1) or (surface > max_surface):
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
            return self.__update_faces(list_faces, box, frame)
        return True

    def cull_faces(self):
        i = 0
        while i < self.face_count():
            face = self.face(i)
            if not face.is_valid():
                self.remove(face)
            else:
                i += 1

    def save_faces(self, dir_out, are_saved_landmarks, is_saved_rectangle):
        for face in self.faces():
            face.save(dir_out, are_saved_landmarks, is_saved_rectangle)

    def write_face_to_frame(self, index, frame, are_saved_landmarks, is_saved_rectangle):
        self.face(index).write_to_frame(frame, are_saved_landmarks, is_saved_rectangle)