from model import handle_call
import json
import cv2

def calc_width(box):
    x_val = (box[1], box[3])
    width = (x_val[1] - x_val[0])
    return width

def calc_middl(box):
    return [(box[3] - box[1]), (box[2] - box[0])]

def calc_distance_cam(adjustment_values, pixl_width):
    #formula: focal length = (Pixelwidth x Distance to Camera) / Width Cone
    # Distance to cam = (focal length x width cone) / pxl width
    return (adjustment_values["focallength"] * adjustment_values["cone_width"]) / pixl_width


# get to predicting image
# run prediction on image
predict_dict, image_path = handle_call(".\\predict", ".\\output")

# label 3 = yellow-cone 1 = blue-cone
# return dict = 
# {boxes: array([[[4.1688204e-01, 5.6460404e-01, 4.7851294e-01, 5.9010553e-01], ... ]]],
# percent of image (y1,x1,y2,x2)
# need image size to calc acutal pixl width -> only x values difference times pixl width
# 'scores': array([[9.9931145e-01, 9.9667001e-01, 9.9516678e-01, 9.7144949e-01, ... ]],
# sort 
# 'classes': array([[3., 3., 3., 1., 1., 1., ..., ]]}

# filter detection dict in only over 60% accuracy
num_preds = 0
for i in predict_dict['scores'][0]:
    if i >= 0.6:
        num_preds += 1
predict_dict_reduced = {"boxes" : [predict_dict["boxes"][0][i] for i in range(num_preds)], 
    "classes" : [predict_dict["classes"][0][i] for i in range(num_preds)]}

# import adjustment values
file = open("adjust.json", "r")
adjustment_values = json.loads(file.read())
file.close()

img = cv2.imread(image_path[0])
h,w, _ = img.shape

# calculating the avg of the width of a cone and it's diameter in order to catch 
# a more general value for the cone width
adjustment_values["cone_width"] = (2**0.5)*(adjustment_values["cone_width"])

# return dict{distance_y_b = .., distance_y = .., distance_b = ...}
return_dict = {}

# if there are one blue and one yellow cone:
if 1.0 in predict_dict_reduced["classes"] and 3.0 in predict_dict_reduced["classes"]:
    print("Blue and yellow cone found!")
    #first is yellow (3) second is blue (1) each first value is width in percent 
    #of img pxl second is postition of middle (x,y) in pxl
    final_cones = [[0,None],[0,None]]
    counter = 0
    # look for biggest blue (1) and yellow (2) cone
    for cone in predict_dict_reduced["classes"]:
        if cone == 3.0:
            width = calc_width(predict_dict_reduced["boxes"][counter])
            if width > final_cones[0][0]:
                final_cones[0][0] = width * w
                x_pos = calc_middl(predict_dict_reduced["boxes"][counter])[0]
                y_pos = calc_middl(predict_dict_reduced["boxes"][counter])[1]
                final_cones[0][1] = (x_pos*w, y_pos*h)
            counter += 1
        if cone == 1.0:
            width = calc_width(predict_dict_reduced["boxes"][counter])
            if width > final_cones[1][0]:
                final_cones[1][0] = width * w
                x_pos = calc_middl(predict_dict_reduced["boxes"][counter])[0]
                y_pos = calc_middl(predict_dict_reduced["boxes"][counter])[1]
                final_cones[1][1] = (x_pos*w, y_pos*h)
            counter += 1
    
    # calculate pixl/meter ratio
    # calculate distance from middle of 1 and 2
    # and calculate distance from camera to 1 and 2

    # meter_pixl avg pixl/meter per cone in order to catch the angle
    meter_pixl = (adjustment_values["cone_width"]/final_cones[0][0] + adjustment_values["cone_width"]/final_cones[1][0]) / 2
    #euclidian distance between the two middles times meter_pixl
    return_dict["distance_y_b"] = (((final_cones[0][1][0] -final_cones[1][1][0])**2 + (final_cones[0][1][1]-final_cones[1][1][1])**2)**0.5) * meter_pixl
    return_dict["distance_y"] = calc_distance_cam(adjustment_values, final_cones[0][0])
    return_dict["distance_b"] = calc_distance_cam(adjustment_values, final_cones[1][0])

# else just calculate distance to existing cone

elif 1.0 in predict_dict_reduced["classes"]:
    print("Only blue cone found!")
    final_cones = [[0,None],[0,None]]
    counter = 0
    for cone in predict_dict_reduced["classes"]:
        width = calc_width(predict_dict_reduced["boxes"][counter])
        if width > final_cones[1][0]:
            final_cones[1][0] = width * w
            x_pos = calc_middl(predict_dict_reduced["boxes"][counter])[0]
            y_pos = calc_middl(predict_dict_reduced["boxes"][counter])[1]
            final_cones[1][1] = (x_pos*w, y_pos*h)
        counter += 1
    
    return_dict["distance_y_b"] = None
    return_dict["distance_y"] = final_cones[0][1]
    return_dict["distance_b"] = calc_distance_cam(adjustment_values, final_cones[1][0])

elif 3.0 in predict_dict_reduced["classes"]:
    print("Only yellow cone found!")
    final_cones = [[0,None],[0,None]]
    counter = 0
    for cone in predict_dict_reduced["classes"]:
        width = calc_width(predict_dict_reduced["boxes"][counter])
        if width > final_cones[0][0]:
            final_cones[0][0] = width * w
            x_pos = calc_middl(predict_dict_reduced["boxes"][counter])[0]
            y_pos = calc_middl(predict_dict_reduced["boxes"][counter])[1]
            final_cones[0][1] = (x_pos*w, y_pos*h)
        counter += 1
    
    return_dict["distance_y_b"] = None
    return_dict["distance_y"] = calc_distance_cam(adjustment_values, final_cones[0][0])
    return_dict["distance_b"] = final_cones[1][1]


print(return_dict)