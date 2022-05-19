# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                              AttendanceProject.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                             https://(address.com) ||
# |                                                                                                      Version 1.0  ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import face_recognition
import time
import colorama
# install dlib in prompt pip install dlib==19.18.0
# ---------------------------------------------------------------------------------------------------------------------|


def compare_face_recognition(dir1, dir2,
                             img_print=True):
    """
    Function in order to make clear how the face_recognition library works.
    It is possible to use two images containing one face each.
    :param dir1: First image directory.
    :param dir2: Second image directory.
    :param img_print: Boolean, whether to display images after code completion.
    :return: None
    """
    print(colorama.Fore.CYAN + "|-------------- Facial Recognition --------------|" + colorama.Style.RESET_ALL)
    initial_process = time.time()  # Time at the moment
    # Image upload:
    img_base = face_recognition.load_image_file(file=dir1)         # Load the image in face_recognition
    img_base = cv2.cvtColor(src=img_base, code=cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    img_test = face_recognition.load_image_file(file=dir2)         # Load the image in face_recognition
    img_test = cv2.cvtColor(src=img_test, code=cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Face locations and encode:
    img_base_loc = face_recognition.face_locations(img=img_base)[0]           # Face location type(list), len(4)
    img_base_enc = face_recognition.face_encodings(face_image=img_base)[0]    # Face Parameters and their distances

    img_test_loc = face_recognition.face_locations(img=img_test)[0]            # Face location type(list), len(4)
    img_test_enc = face_recognition.face_encodings(face_image=img_test)[0]     # Face Parameters and their distances

    # Compare the two images:
    # It is necessary to compare to encode of each one of them:
    result = face_recognition.compare_faces(known_face_encodings=[img_base_enc],
                                            face_encoding_to_check=img_test_enc)  # [list], object(list)

    face_distance = face_recognition.face_distance(face_encodings=[img_base_enc],
                                                   face_to_compare=img_test_enc)  # [list], object(list)

    final_process = time.time()  # Time at the moment

    # LOG ##############################################################################################################
    print("Compare faces |---------------------------",
          (colorama.Fore.GREEN if result else colorama.Fore.RED), result,
          colorama.Style.RESET_ALL)
    print("Face distance |-----------------------------",
          (colorama.Fore.GREEN if result else colorama.Fore.RED),
          round(face_distance[0], 2),
          colorama.Style.RESET_ALL)
    print("Runtime       |----------------------" +
          colorama.Fore.YELLOW +
          " {a} seconds".format(a=round(final_process - initial_process, 2)) +
          colorama.Style.RESET_ALL)
    ####################################################################################################################

    if img_print:
        # Rectangle printer:
        cv2.rectangle(img=img_base,
                      pt1=(img_base_loc[3], img_base_loc[0]),
                      pt2=(img_base_loc[1], img_base_loc[2]),
                      color=(0, 255, 0),
                      thickness=1)

        cv2.rectangle(img=img_test,
                      pt1=(img_test_loc[3], img_test_loc[0]),
                      pt2=(img_test_loc[1], img_test_loc[2]),
                      color=(0, 255, 0),
                      thickness=1)

        # Result and face distance printer:
        cv2.putText(img=img_test,
                    text=f'{result} - face distance: {round(face_distance[0], 2)}',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255, 255, 255),
                    thickness=1)

        # Show the image:
        cv2.imshow("img_base", img_base)
        cv2.imshow("img_test", img_test)
        cv2.waitKey(0)  # Frame update


if __name__ == "__main__":
    compare_face_recognition(dir1='imageDatabase/Elon Musk.jpg',
                             dir2='imageDatabase/Elon Musk (1).jpg',
                             img_print=True)

