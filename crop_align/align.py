import cv2
import numpy as np


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

class MyFaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth


    def align(self, image, shape):
        # convert the landmark (x, y)-coordinates to a NumPy array
        # shape = self.predictor(gray, rect)
        # shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        # (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        # leftEyePts = shape[lStart:lEnd]
        # rightEyePts = shape[rStart:rEnd]
        leftEyePts = shape[:, 36:42]
        rightEyePts = shape[:, 42:48]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=1).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=1).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


class MyFaceAligner_RAF:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth


    def align(self, image, shape):
        # convert the landmark (x, y)-coordinates to a NumPy array
        # shape = self.predictor(gray, rect)
        # shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        # (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        # leftEyePts = shape[lStart:lEnd]
        # rightEyePts = shape[rStart:rEnd]
        # leftEyePts = shape[:, 36:42]
        # rightEyePts = shape[:, 42:48]

        # compute the center of mass for each eye
        leftEyeCenter = shape[:, 0].astype("int")
        rightEyeCenter = shape[:, 1].astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/scratch_net/biwidl213/emotion/shape_predictor_68_face_landmarks.dat")
#
# image = cv2.imread('img.jpg')
# fa = FaceAligner(predictor, desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)
#
# faceAligned = fa.align(image, shape)
# cv2.imwrite("Aligned.jpg", faceAligned)