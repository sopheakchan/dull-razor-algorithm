import cv2 as cv
import numpy as np
import os

# define input and output folders path
input_folder = 'skincancer_metadata' # Folder containing class , subfolders with images
output_folder = 'preprocessed_metadata_images' # Folder to save processed images

#ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
#function to apply dull razor algorithm to an image
def dull_razor(image):
    # convert to grayscale
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # blackhat filtering (shape and size)
    kernel = cv.getStructuringElement(1,(11,11))
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
    
    # gaussian blue
    bhg = cv.GaussianBlur(blackhat, (3,3), cv.BORDER_DEFAULT)
    
    # binary tresholding (mask)
    _, mask = cv.threshold(bhg, 10, 255, cv.THRESH_BINARY)
    
    # apply dilation before inpainting
    kernel_dilate = np.ones((9,9), np.uint8)
    dilated_mask = cv.dilate(mask, kernel_dilate, iterations=1)
    
    # inpainting
    dst = cv.inpaint(image, dilated_mask, 6, cv.INPAINT_TELEA)
    
    return dst

# process all images in all subdirectories
for root , dirs , files in os.walk(input_folder):
    for file in files:
        if file.endswith('.jpg'):
            input_path = os.path.join(root, file)
            
            # read image
            image = cv.imread(input_path, cv.IMREAD_COLOR)
            
            # apply dull razor algorithm
            cleaned_image = dull_razor(image)
            
            # create corresponding output path
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder,relative_path)
            
            # ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # save the cleaned image 
            cv.imwrite(output_path,cleaned_image)
            
            print(f"Processed and saved : {output_path}")
        
print("All images processed successfully")