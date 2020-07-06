import sys
import json
import cv2
from model import handle_call


def start():
    stored = dict()
    try:
        file = open("adjust.json", "r")
        stored = json.loads(file.read())
        file.close()
    except:
        None
    answer = "n"
    # print(stored['Width_cone'])
    # check if there is already an ajustment file
    if stored != {}:
        print("Cone width:", stored['cone_width'])
        print("Focallength:", stored['focallength'])
        answer = input("Is this correct? Y/N").lower()
    # if this file isn't correct or not existing
    if answer == "n":
        answer2 = input("Adjusting? Y/N").lower()
        if answer2 == "y":
            cone_width = input("Cone width in Meter (e.g. 0.23):")
            print("Check documentary for correct image setup ... !")
            PATH_TO_ADJUSTMENT_IMG = input("Path to the adjustment image directory (default .\\adjust):")
            if PATH_TO_ADJUSTMENT_IMG == "":
                PATH_TO_ADJUSTMENT_IMG = ".\\adjust"
            adjust_dict = calc_values(handle_call(PATH_TO_ADJUSTMENT_IMG, ".\\output"), float(cone_width))
            
            # store values to adjust.json
            a_file = open("adjust.json", "w")
            json.dump(adjust_dict, a_file)
            a_file.close()
    print("Adjustment succesfull, check the adjust.json file")

#formula: focal length = (Pixelwidth x Distance to Camera) / Width Cone
# label 3 = yellow-cone  = blue-cone
# return dict = 
# {boxes: array([[[4.1688204e-01, 5.6460404e-01, 4.7851294e-01, 5.9010553e-01], ... ]]],
# percent of image (y1,x1,y2,x2)
# need image size to calc acutal pixl width -> only x values difference times pixl width
# 'scores': array([[9.9931145e-01, 9.9667001e-01, 9.9516678e-01, 9.7144949e-01, ... ]]],
# sort 
# 'classes': array([[3., 3., 3., 1., 1., 1., ..., ]]}

def calc_values(model_return, cone_width):
    #assumption: the adjusting picture is taken as described and also 
    # I trust in the Model so it will detect the big yellow-cone with high accuracy

    predict_dict, PATH_TO_ADJUSTMENT_IMG = model_return

    x_val = (predict_dict["boxes"][0][0][1], predict_dict["boxes"][0][0][3])
    img = cv2.imread(PATH_TO_ADJUSTMENT_IMG[0])
    h,w, _ = img.shape
    pixl_width = (x_val[1] - x_val[0]) * w
    #The Camera has to be placed 1m distant to the cone and 0.2m high
    distance_to_camera = (1 + 0.2**2)**0.5
    focallength = (pixl_width * distance_to_camera) / cone_width

    return {"focallength" : focallength, "cone_width" : cone_width}


if __name__ == "__main__":
   start()
