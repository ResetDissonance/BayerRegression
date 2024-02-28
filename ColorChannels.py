import numpy as np
import cv2
import random
import sys
from enum import Enum
import Node



DEBUG_FLAG = int(sys.argv[1])
colors = ["blue","green","red"]
pictures = ["0900x4.png","Climber.png","Flower.png","Oranges.png","Trees.png","CameraMan.png"]

#Returns GrayScale Version of Bayer Patches
def GetBayerPatches(image, patch_size, pattern,num_patches):
    #Set up patch size and pattern
    offset = int(patch_size/2)
    first = int(pattern[0])
    second = int(pattern[1])
    third = int(pattern[2])
    fourth = int(pattern[3])
    
    
    if(DEBUG_FLAG):
        print()
        #print(image[random_row-offset:random_row+offset+1,random_col-offset:random_col+offset+1])
        
    middle_pixels = np.array([np.zeros(3) for _ in range(num_patches)])
    #patches = np.array(([[np.zeros(3) for _ in range(5)] for _ in range(5)]))
    patches = np.array([[[np.zeros(3) for _ in range(patch_size)] for _ in range(patch_size)] for _ in range(num_patches)])
    
    for file in pictures:
        completed_pixels = []
        image = cv2.imread(file)
        number_of_pixels = (len(image)-2*offset-3)*(len(image[0])-2*offset-3)
        #print("Training on ", file)
        for total in range(0,num_patches):
            if(len(completed_pixels)>=number_of_pixels):
                break
            random_row = random.randint(offset,len(image) - offset - 1)
            random_col = random.randint(offset,len(image[0]) - offset - 1)
            while([random_row,random_col] in completed_pixels):
                # print("COMPLETED PIXELS",len(completed_pixels))
                # print("TOTAL PIXELS: ",number_of_pixels)
                random_row = random.randint(offset,len(image) - offset - 1)
                random_col = random.randint(offset,len(image[0]) - offset - 1)
            completed_pixels.append([random_row,random_col])
            # random_row = 113
            # random_col = 231
            curr_middle = image[random_row, random_col].copy()
            curr_patch = image[random_row-offset:random_row+offset+1,random_col-offset:random_col+offset+1].copy()
            middle_pixels[total] = image[random_row, random_col].copy()
            patches[total] = image[random_row-offset:random_row+offset+1,random_col-offset:random_col+offset+1].copy()
            
            
            #print("GETTING {} pattern from a random pixel at".format(str(pattern)), random_row, random_col)
            
            
            #cv2.imshow("before", patches)
            
            for j in range(0,len(patches[0])):
                for i in range(0,len(patches[0])):
                    if(j%2==0):
                        if(i%2 == 0):
                            patches[total][j][i] = patches[total][j,i,first]
                        else:
                            patches[total][j][i] = patches[total][j,i,second]
                    else:
                        if(i%2 == 0):
                            patches[total][j][i] = patches[total][j,i,third]
                        else:
                            patches[total][j][i] = patches[total][j,i,fourth]
            
        #cv2.imshow("after GrayScaleBAYER", patches)
    
    return patches, middle_pixels

def Create_Bayer(pattern,name):
    image = cv2.imread(name)
    color_channel = image.copy()
    
    first = int(pattern[0])
    second = int(pattern[1])
    third = int(pattern[2])
    fourth = int(pattern[3])
    for j in range(0,len(image)):
        for i in range(0,len(image[0])):
            if(j%2==0):
                if(i%2 == 0):
                    color_channel[j,i,:] = image[j,i,first]
                else:
                    color_channel[j,i,:] = image[j,i,second]
            else:
                if(i%2 == 0):
                    color_channel[j,i,:] = image[j,i,third]
                else:
                    color_channel[j,i,:] = image[j,i,fourth]
    
    cv2.imwrite('BAYER_{}'.format(name),color_channel)
    return color_channel
 
def Show_Bayer(pattern, image):
    color_channel = image.copy()
    
    first = int(pattern[0])
    second = int(pattern[1])
    third = int(pattern[2])
    fourth = int(pattern[3])
    for j in range(0,len(image)):
        for i in range(0,len(image[0])):
            if(j%2==0):
                if(i%2 == 0):
                    color_channel[j,i,:] = 0
                    color_channel[j,i,first] = image[j,i,first]
                else:
                    color_channel[j,i,:] = 0
                    color_channel[j,i,second] = image[j,i,second]
            else:
                if(i%2 == 0):
                    color_channel[j,i,:] = 0
                    color_channel[j,i,third] = image[j,i,third]
                else:
                    color_channel[j,i,:] = 0
                    color_channel[j,i,fourth] = image[j,i,fourth]
    
    cv2.imshow("BAYER",color_channel)
    return color_channel

def Convert_Patches_to_LLS(patch_size, ground_truth, patch, pattern):
    LLS_row_size = len(patch)
    if(DEBUG_FLAG):
        print("LLS rows is",LLS_row_size)
    LLS_col_size = patch_size * patch_size
    
    #Bayer Arrays created
    y1 = np.zeros(LLS_row_size)
    y2 = np.zeros(LLS_row_size)
    A = np.zeros((LLS_row_size,LLS_col_size))
    A1 = np.zeros((LLS_row_size,LLS_col_size))
    A2 = np.zeros((LLS_row_size,LLS_col_size))
    if(DEBUG_FLAG):
        print(len(A))
        # print(patch)
        # print(patch[0][1][0])
        # print(len(A))
        print("PATCH TO CONVERT:",patch[0])
        print("Ground truth: ", ground_truth[0])
    for big_row in range(len(A)):
        y1[big_row],y2[big_row] = Determine_Missing_Elements(ground_truth[big_row],patch_size,pattern)
        for row in range(len(patch[big_row])):
            multiplier = row*patch_size
            for col in range(len(patch[big_row][0])):
                A[big_row][multiplier+col] = patch[big_row][row][col][0]
        A1[big_row] = Clean_Up_Matrix(1,A[big_row],pattern,patch_size)
        A2[big_row] = Clean_Up_Matrix(2,A[big_row],pattern,patch_size)
    
    
    #print("A:",A,"y1",y1,"y2",y2)
    return A,A1,A2,y1,y2

def Clean_Up_Matrix(type, A_row, pattern,patch_size):
    middle_row = patch_size//2
    A_op = A_row.copy()
    if(type==1):
        if(pattern == '2110'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==0):
                            A_op[multiplier+col] = 0
        elif(pattern == '1021'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
        elif(pattern == '1201'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
            
        elif(pattern == '0112'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==0 and col%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
    if(type==2):
        if(pattern == '2110'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0 and col%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
        elif(pattern == '1021'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
        elif(pattern == '1201'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
                        if(col%2==1):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
            
        elif(pattern == '0112'):
            if(middle_row%2==1):
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
                        if(row%2==0 and col%2==0):
                            A_op[multiplier+col] = 0
            else:
                for row in range(0,patch_size):
                    multiplier = row*patch_size
                    for col in range(0,patch_size):
                        if(row%2==0):
                            A_op[multiplier+col] = 0
                        if(row%2==1 and col%2==1):
                            A_op[multiplier+col] = 0
    return A_op

def LLS(A,y):
    pinv = np.linalg.pinv(A)
    alpha = pinv.dot(y)
    # print(np.dot(A,alpha))
    return alpha
    
    
def Determine_Missing_Elements(ground_truth,patch_size, pattern):
    middle_row = patch_size//2
    if(middle_row%2==1):
        #print("MIDDLE PIXEL VALUE",colors[int(pattern[3])])
        match int(pattern[3]):
            case 0:
                y1 = ground_truth[1]
                y2 = ground_truth[2]
            case 1:
                y1 = ground_truth[0]
                y2 = ground_truth[2]
            case 2:
                y1 = ground_truth[0]
                y2 = ground_truth[1]
    elif(middle_row%2==0):
        #print("MIDDLE PIXEL VALUE",colors[int(pattern[0])])
        match int(pattern[0]):
            case 0:
                y1 = ground_truth[1]
                y2 = ground_truth[2]
            case 1:
                y1 = ground_truth[0]
                y2 = ground_truth[2]
            case 2:
                y1 = ground_truth[0]
                y2 = ground_truth[1]
    else:
        print("ERROR")
    return y1,y2

def Assign_Missing_Elements(pattern, estimate,row,col,y1,y2,patch_size):
    middle_row = patch_size//2
    y1 = y1[0]
    y2 = y2[0]
    if(middle_row%2==1):
        #print("MIDDLE PIXEL VALUE",colors[int(pattern[3])])
        match int(pattern[3]):
            case 0:
                estimate[1] = y1
                estimate[2] = y2
            case 1:
                estimate[0] = y1
                estimate[2] = y2
            case 2:
                estimate[0] = y1
                estimate[1] = y2
    elif(middle_row%2==0):
        #print("MIDDLE PIXEL VALUE",colors[int(pattern[0])])
        match int(pattern[0]):
            case 0:
                estimate[1] = y1
                estimate[2] = y2
            case 1:
                estimate[0] = y1
                estimate[2] = y2
            case 2:
                estimate[0] = y1
                estimate[1] = y2
    else:
        print("ERROR")
    return estimate


def Train_Coefficient_Matrices(image, num_patches, patch_size,bayer_patterns):
    RGGB_coeff1 = np.zeros(patch_size*patch_size)
    RGGB_coeff2 = np.zeros(patch_size*patch_size)
    
    GBRG_coeff1 = np.zeros(patch_size*patch_size)
    GBRG_coeff2 = np.zeros(patch_size*patch_size)
    
    GRBG_coeff1 = np.zeros(patch_size*patch_size)
    GRBG_coeff2 = np.zeros(patch_size*patch_size)
    
    BGGR_coeff1 = np.zeros(patch_size*patch_size)
    BGGR_coeff2 = np.zeros(patch_size*patch_size)
    
    for i in bayer_patterns:
        array_of_patches,middle_pixels = GetBayerPatches(image,patch_size,i,num_patches)
        A,A1,A2,y1,y2 = Convert_Patches_to_LLS(patch_size,middle_pixels,array_of_patches,i)
        alpha1 = LLS(A1,y1)
        alpha2 = LLS(A2,y2)
        match i:
            case '2110':
                RGGB_coeff1 = alpha1
                RGGB_coeff2 = alpha2
                Input_A = A
                Output_Y = y1
            case '1021':
                GBRG_coeff1 = alpha1
                GBRG_coeff2 = alpha2
            case '1201':
                GRBG_coeff1 = alpha1
                GRBG_coeff2 = alpha2
            case '0112':
                BGGR_coeff1 = alpha1
                BGGR_coeff2 = alpha2
    return RGGB_coeff1,RGGB_coeff2,GBRG_coeff1,GBRG_coeff2,GRBG_coeff1,GRBG_coeff2,BGGR_coeff1,BGGR_coeff2

def Demosaic_Image(mosaic, initial_pattern,bayer_patterns,patch_size,image):
    start_node = Node.patterns.get_node(initial_pattern)

    offset = patch_size//2
    
    curr_row_pattern = start_node
    curr_col_pattern = start_node
    
    start_row_pixel = offset
    end_row_pixel = len(mosaic) - offset - 1
    
    start_col_pixel = offset
    end_col_pixel = len(mosaic[0]) - offset - 1
    
    for row in range(start_row_pixel,end_row_pixel):
        if(row != start_row_pixel):
            curr_row_pattern = curr_row_pattern.down
            curr_col_pattern = curr_row_pattern
        for col in range(start_col_pixel,end_col_pixel):
            if(col != start_col_pixel):
                curr_col_pattern = curr_col_pattern.right
            curr_patch = mosaic[row-offset:row+offset+1,col-offset:col+offset+1].copy()
            original_patch = image[row-offset:row+offset+1,col-offset:col+offset+1].copy()
            A = Construct_A_Matrix(curr_patch,patch_size)
            alpha1,alpha2 = Get_Coeff_Matrices(curr_col_pattern.data)
            y1 = np.dot(A,alpha1)
            y2 = np.dot(A,alpha2)
            mosaic[row][col] = Assign_Missing_Elements(curr_col_pattern.data,mosaic[row][col],row,col,y1,y2,patch_size)
    return mosaic
            

def Get_Coeff_Matrices(pattern):
    match pattern:
        case '2110':
            return RGGB_coeff1,RGGB_coeff2
        case '1021':
            return GBRG_coeff1,GBRG_coeff2
        case '1201':
            return GRBG_coeff1,GRBG_coeff2
        case '0112':
            return BGGR_coeff1,BGGR_coeff2
            
def Construct_A_Matrix(patch,patch_size):
    A = np.zeros((1,patch_size*patch_size))
    for row in range(len(patch)):
            multiplier = row*patch_size
            for col in range(len(patch[0])):
                A[0][multiplier+col] = patch[row][col][0]
    return A

def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 0.5)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 0.5)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

def LAB_Histogram_Equalization(img_in):
    lab = cv2.cvtColor(img_in,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l_new = cv2.equalizeHist(l)
    lab_new = cv2.merge([l_new,a,b])
    img_new = cv2.cvtColor(lab_new,cv2.COLOR_LAB2BGR)
    
    return img_new

def Histogram_Equalization(img_in):
# segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
# calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
# mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
# merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
    img_out = cv2.merge((img_b, img_g, img_r))
# validation
    # equ_b = cv2.equalizeHist(b)
    # equ_g = cv2.equalizeHist(g)
    # equ_r = cv2.equalizeHist(r)
    # equ = cv2.merge((equ_b, equ_g, equ_r))
    
    return img_out


def mse(imageA, imageB):
	
	mse = np.sqrt(np.sum(((imageA - imageB)**2).mean(axis=None)))
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return mse
        


RGGB = '2110'
GBRG = '1021'
GRBG = '1201'
BGGR = '0112'



#Create_Bayer(RGGB,"Tools.png")
mosaic = cv2.imread("kodim05.png")
image = cv2.imread("0900x4.png")
# matlab_de = cv2.imread("Demosaic_0900x4.png")
bayer_patterns = [RGGB,GBRG,GRBG,BGGR]
num_patches = 5000
patch_size = 5


RGGB_coeff1,RGGB_coeff2,GBRG_coeff1,GBRG_coeff2,GRBG_coeff1,GRBG_coeff2,BGGR_coeff1,BGGR_coeff2 = Train_Coefficient_Matrices(image,num_patches,patch_size,bayer_patterns)

final_img = Demosaic_Image(mosaic,RGGB,bayer_patterns,patch_size,image)
cv2.imshow("DEMOSAIC",final_img)

# HE_ogimg = Histogram_Equalization(final_img)
# cv2.imshow("HE OG img",HE_ogimg)

# LAB_HE = LAB_Histogram_Equalization(final_img)
# cv2.imshow("LAB_HE",LAB_HE)

# white_balanced_img = GW_white_balance(final_img)
# cv2.imshow("WHITE",white_balanced_img)

# HE_WB_img = GW_white_balance(HE_ogimg)
# cv2.imshow("HE_WB_img",HE_WB_img)

# print("Number of Patches taken per training image: ", num_patches, "Patch Size: ",patch_size)
# print("RMSE ERROR of matlab demosaiced and real",mse(matlab_de,image))
# print("RMSE ERROR of my demosaic and real",mse(final_img,image))
# print("RMSE ERROR between the two histogram equalization algorithms",mse(LAB_HE,HE_ogimg))
# # print("RMSE ERROR of HE demosaic and real",mse(HE_ogimg,image))
# print("RMSE ERROR of WB&HE demosaic and real",mse(HE_WB_img,image))


dark = cv2.imread("10087_00_30s.jpg")
cv2.imshow("dark",dark)
cv2.imshow("HE",Histogram_Equalization(dark))
# print("RGGB_coeff1",RGGB_coeff1,"\nRGGB_coeff2",RGGB_coeff2)

# print("GBRG_coeff1",GBRG_coeff1,"\nGBRG_coeff2",GBRG_coeff2)

# print("GRBG_coeff1",GRBG_coeff1,"\nGRBG_coeff2",GRBG_coeff2)

# print("BGGR_coeff1",BGGR_coeff1,"\nRGGB_coeff2",BGGR_coeff2)



cv2.waitKey(0)