## this code is inspired from:
## https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys, getopt
from model import handle_call 

def main(argv):
    path_to_images = ''
    output_path = ""
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["path_image=", "output_path="])
    except getopt.GetoptError:
        print ('predict.py -i <path_to_image_directory> -o <output_path_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('predict.py -i <path_to_image_directory> -o <output_path>')
            sys.exit()
        elif opt in ("-i", "--path_image"):
            path_to_images = arg
        elif opt in ("-o", "--output_path"):
            output_path = arg

    print(path_to_images, output_path)

    if path_to_images == "":
        path_to_images = r".\images"
    
    if output_path == "":
        output_path = r".\output"

    ans = handle_call(path_to_images, output_path)
    print("check your specified output directory or default .\\output")

if __name__ == "__main__":
   main(sys.argv[1:])