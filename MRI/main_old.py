import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import cv2
import os
from os.path import isfile, join
from os import listdir
import argparse
import keras
from keras import optimizers, callbacks, models
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import loss_functions
import utils
import model_storage
import cv2
from PIL import Image, ImageEnhance
import PIL
from skimage import measure
from skimage import exposure
from sklearn.preprocessing import QuantileTransformer
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


###########
#
#   Generate Patches and Store them in A Numpy File
#
###########
def SharpTheImage(img, verbose=False, type_of_sharpness="Classic"):
    if type_of_sharpness == 'Classic':
        gaussian_1 = cv2.GaussianBlur(img, (9,9),10.0) #(9,9), 10.0)
        if verbose:
            print('gaussian_1')
            plt.imshow((gaussian_1))
            plt.show()
        img_copy = cv2.addWeighted(img, 1.5, gaussian_1, -0.5, 0, img) 
        if verbose:
            print('img_copy')
            plt.imshow((img_copy))
            plt.show()
        return img_copy
    elif type_of_sharpness == 'TwoLevel':
        from scipy import ndimage
        blurred_f = ndimage.gaussian_filter(img, 2)
        if verbose:
            plt.imshow((blurred_f))
            plt.show()
                
        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        if verbose:
            print('filter_blurred_f')
            plt.imshow((filter_blurred_f))
            plt.show()
        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        if verbose:
            print('sharpened')
            plt.imshow((sharpened))
            plt.show()
    elif type_of_sharpness == 'EDGE_ENHANCE':
        kernel = np.array(([-1, -1, -1],[-1, 15, -1],[-1, -1, -1]), dtype='int')
        sharpened = cv2.filter2D(img, -1, kernel)
        if verbose:
            plt.imshow(sharpened)
            plt.show()

    elif type_of_sharpness== 'Convolute':
        kernel = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
        #convoleOutput = convolve(gray, kernel)
        sharpened = cv2.filter2D(img, -1, kernel)
        if verbose:
            plt.imshow(sharpened)
            plt.show()
    else:
        sharpened = img
    
    return sharpened

def slice_mri(mri_filename="Case00.mhd", mask_filename="Case00_segmentation.mhd",max_shape=(512,512), verbose=False, short=False):
    '''
    mri_filename = MRI file
    mask_filename = Mask file
    max_shape = (512,512)
    verbose = Boolean
    short = Boolean
    quantile_normalize = Boolean
    '''
    #load MRI image and mask
    image = sitk.ReadImage(mri_filename)
    '''
    #image = sitk.AdaptiveHistogramEqualization(image)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)

    image = sitk.Cast( image, sitk.sitkFloat32 )
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    corrector.SetMaximumNumberOfIterations( [1 *numberFittingLevels]  )
    
    output = corrector.Execute( image, maskImage )
    rescale = sitk.RescaleIntensityImageFilter()
    output = rescale.Execute(output)
    #output = sitk.Normalize(output)
    '''
    ct_scan = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(mask_filename)
    ct_mask = sitk.GetArrayFromImage(mask)
    
    ct_scan_resized = np.zeros((ct_scan.shape[0], max_shape[0], max_shape[1]))
    ct_mask_resized = np.zeros((ct_mask.shape[0], max_shape[0], max_shape[1]))
    ic=50
    verbose=True
    from skimage import exposure
    for index, img in enumerate(ct_scan):
        img_copy = img.copy()
        #print(np.max(img_copy))
        #img_copy = img_copy / np.max(img_copy)
        if img.shape==max_shape:
            #Normalize the MRI images
            #img_copy = np.nan_to_num(img_copy)
            if verbose:
                print('raw data')
                print(np.min(img_copy))
                print(np.max(img_copy))
                plt.imshow((img_copy))
                plt.show()
            #
            #for i in range(3,20,2):
            #    for j in range(3,20):
            #        print(i,j)
            '''
            gaussian_1 = cv2.GaussianBlur(img_copy, (7,7),9) #(9,9), 10.0)
            #        plt.imshow((gaussian_1))
            #        plt.show()
            print('sharpend the data')
            img_copy = cv2.addWeighted(img_copy, 1.5, gaussian_1, -0.5, 0, img_copy) 
            plt.imshow((img_copy))
            plt.show()
            '''
            
            sharpened = SharpTheImage(img_copy,verbose)
            if verbose:
                print('sharpened')
                plt.imshow((sharpened))
                plt.show()
            
            img_copy = exposure.equalize_adapthist(sharpened, clip_limit=0.05, nbins=1000)#,nbins=100)
            
            if verbose:
                print('EqAH')
                plt.imshow((img_copy))
                plt.show()
            
            img_copy = np.nan_to_num(img_copy)
            img_copy[np.isnan(img_copy)] = 0.
            #Add image
            ct_scan_resized[index,:,:] = img_copy.copy()
            ct_mask_resized[index] = ct_mask[index].copy()
        else:
            if verbose:
                print('img.shape',img.shape)
                print('mask.shape',ct_mask[index].shape)
                print(np.min(img_copy))
                print(np.max(img_copy))
                plt.imshow((img_copy))
                plt.show()
            
            #Normalize the MRI images
            #img_copy = np.nan_to_num(img_copy) 
            '''
            gaussian_1 = cv2.GaussianBlur(img_copy, (9,9), 10.0)
            img_copy = cv2.addWeighted(img_copy, 1.5, gaussian_1, -0.5, 0, img_copy) 
            '''
            sharpened = SharpTheImage(img_copy,verbose)
            img_copy = exposure.equalize_adapthist(sharpened, clip_limit=0.05, nbins=1000)#,nbins=100)
            img_copy = Image.fromarray(img_copy)
            '''
            width, height = img_copy.size   # Get dimensions
            
            new_width = max_shape[0]
            new_height = max_shape[1]
            print(new_width,new_height)
            print(width, height)
            left = abs((new_width-width)/2)
            top = abs((new_height-height)/2)
            right = abs((width + new_width)/2)
            bottom = abs((height + new_height)/2)

            print('cropped...')
            img_copy = img_copy.crop((left, top, right, bottom))
            '''
            img_copy = img_copy.resize(max_shape, resample=PIL.Image.LANCZOS)
            img_copy = np.array(img_copy)
            #img_copy = img_copy[0:new_height,0:new_width]
            
            if verbose:
                plt.imshow((img_copy))
                plt.show()
            #img_copy[np.isnan(img_copy)] = 0.
            #img_copy = Image.fromarray(img_copy)
            
            if verbose:
                plt.imshow((img_copy))
                plt.show()
            #img_copy = np.array(img_copy)
            
            #Mask
            mask_copy = Image.fromarray(ct_mask[index])
            mask_copy = mask_copy.resize(max_shape)
            mask_copy = np.array(mask_copy)
            #Add image
            ct_scan_resized[index,:,:] = img_copy.copy()
            ct_mask_resized[index] = mask_copy.copy()
            
    ct_mask = ct_mask_resized
    ct_scan = ct_scan_resized
    if verbose:
        print('ct_mask',ct_mask.shape)
        print('ct_scan',ct_scan.shape)
        plt.imshow((ct_mask[0]))
        plt.show()
        plt.imshow((ct_scan[0]))
        plt.show()
    if short:
        return ct_scan, ct_mask
    

    #Label contour
    ct_mask_data = ct_mask.copy()
    for n in range(ct_mask.shape[0]):    
        label_boolean = ct_mask_data[n]
        label_boolean = label_boolean.astype(np.uint8)
        label_boolean = label_boolean *255
        ret, thresh = cv2.threshold(label_boolean,127,255,0)
        if verbose:
            plt.imshow(thresh)
            plt.show()
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        points_list = []
        for contour in contours:
            points = list(contour)
            points.append(points[0])
            points = np.array(points)
            points = points.astype(int)
            points_list.append(points)
        for point_lst in points_list:
            cv2.polylines(ct_mask_data[n],[point_lst],True,2,3)
            
        if verbose:
            plt.imshow((ct_mask_data[n]))
            plt.show()
    return ct_scan, ct_mask_data

def GetROIOnly(mri_img, mri_mask, size_window, verbose=False):
    print("Clipping only ROI...")
    mri_img_new = []#np.zeros((len(mri_img), size_window[0], size_window[1]))
    mri_mask_new = [] #np.zeros((len(mri_mask), size_window[0], size_window[1]))

    mri_img_new_0_0 = []#np.zeros((len(mri_img), size_window[0], size_window[1]))
    mri_mask_new_0_0 = [] #np.zeros((len(mri_mask), size_window[0], size_window[1]))
    index = 0
    if (mri_img[0].shape[0], mri_img[0].shape[0]) == size_window:
        print('same size window')
        for (mri_img_single, mri_mask_single) in zip(mri_img, mri_mask):
            if np.count_nonzero(mri_mask_single)>0:
                mri_img_new_0_0.append(mri_img_single)
                mri_mask_new_0_0.append(mri_mask_single)

        mri_img_new,mri_mask_new= np.array(mri_img_new_0_0), np.array(mri_mask_new_0_0)
        print('mri_img_new',mri_img_new.shape)
        print('mri_mask_new',mri_mask_new.shape)
        return mri_img_new,mri_mask_new

    for (mri_img_single, mri_mask_single) in zip(mri_img, mri_mask):
        if np.count_nonzero(mri_mask_single)>0:
            if verbose:
                print("fired 1")
            lbls = measure.label(mri_mask_single>0)
            for lbl in measure.regionprops(lbls):
                img = lbl.image
                min_row, min_col, max_row, max_col  = lbl.bbox
                height, width = max_row - min_row, max_col - min_col
                if verbose:
                    print('image shape', img.shape)
                diff_y = int(round((size_window[0] - img.shape[0]) * 0.5))
                diff_x = int(round((size_window[1] - img.shape[1]) * 0.5))

                
                if diff_x <0:
                    diff_x = 0
                if diff_y < 0:
                    diff_y = 0
                if verbose:
                    print(diff_x, diff_y)
                    print('mri image')
                    plt.imshow(mri_img_single[min_row:max_row, min_col:max_col])
                    plt.show()
                    print('mri mask 1')
                    plt.imshow(img)
                    plt.show()
                    print('mri mask 2')
                    plt.imshow(mri_mask_single[min_row:max_row, min_col:max_col])
                    plt.show()
                    #print(mri_img_new.shape)
                x_min = min_col - diff_x
                y_min = min_row - diff_y
                x_max = x_min + size_window[1] 
                y_max = y_min + size_window[0]
                if x_min <0:
                    x_min = 0
                    x_max = size_window[1]
                if y_min <0:
                    y_min = 0
                    y_max = size_window[0]
                img = np.zeros(size_window)
                img_copy = mri_img_single[y_min:y_max, x_min:x_max]
                img[0:img_copy.shape[0],0:img_copy.shape[1]] = img_copy
                mri_img_new.append(img.copy())
                if verbose:
                    plt.imshow(img)
                    plt.show()
                
                img = np.zeros(size_window)
                img_copy = mri_img_single[min_row:min_row+size_window[0], min_col:min_col+size_window[1]]
                img[0:img_copy.shape[0],0:img_copy.shape[1]] = img_copy
                mri_img_new.append(img.copy())
                if verbose:
                    plt.imshow(img)
                    plt.show()
                
                mask = np.zeros(size_window)
                mask_copy = mri_mask_single[y_min:y_max, x_min:x_max]
                mask[0:mask_copy.shape[0],0:mask_copy.shape[1]] = mask_copy
                mri_mask_new.append(mask.copy())
                
                mask = np.zeros(size_window)
                mask_copy = mri_mask_single[min_row:min_row+size_window[0], min_col:min_col+size_window[1]]
                mask[0:mask_copy.shape[0],0:mask_copy.shape[1]] = mask_copy
                mri_mask_new.append(mask.copy())
                
                if verbose:
                    print('mri mask 3')
                    plt.imshow(mri_mask_new[index])
                    plt.show()
                    print('mri img 3')
                    plt.imshow(mri_img_new[index])
                    plt.show()
                    print(diff_y,diff_x)
        else:
            if verbose:
                print("fired 2")
            diff_y = int(round((mri_img_single.shape[0] - size_window[0]) * 0.5))
            diff_x = int(round((mri_img_single.shape[1] - size_window[1]) * 0.5))
            
            if verbose:
                print(diff_y, diff_x)
                plt.imshow(mri_img_single[diff_y:diff_y+size_window[0],diff_x:diff_x+size_window[1]])
                plt.show()
                plt.imshow(mri_mask_single[diff_y:diff_y+size_window[0],diff_x:diff_x+size_window[1]])
                plt.show()
            '''
            img = np.zeros(size_window)
            mask = np.zeros(size_window)
            img_copy = mri_img_single[diff_y:diff_y+size_window[0],diff_x:diff_x+size_window[1]]
            mask_copy = mri_mask_single[diff_y:diff_y+size_window[0],diff_x:diff_x+size_window[1]]
            img[0:img_copy.shape[0],0:img_copy.shape[1]] = img_copy
            mask[0:mask_copy.shape[0],0:mask_copy.shape[1]] = mask_copy
            mri_img_new.append(img)
            mri_mask_new.append(mask)
            '''
            
        index += 1
    
    mri_img_new_,mri_mask_new_= np.array(mri_img_new), np.array(mri_mask_new)
    '''
    for img, mask in zip(mri_img_new_,mri_mask_new_):
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst_img = cv2.warpAffine(img,M,(cols,rows))
        dst_mask = cv2.warpAffine(mask,M,(cols,rows))
        mri_img_new.append(dst_img)
        mri_mask_new.append(dst_mask)
        if verbose:
            plt.imshow(dst_img)
            plt.show()
            plt.imshow(dst_mask)
            plt.show()
    '''
    b = ImageDataGenerator()
    mri_img_new_h = b.apply_transform(x=mri_img_new_,transform_parameters= {'flip_horizontal':True})
    mri_mask_new_h = b.apply_transform(x=mri_mask_new_,transform_parameters= {'flip_horizontal':True})
    
    for img, mask in zip(mri_img_new_h,mri_mask_new_h):
        mri_img_new.append(img)
        mri_mask_new.append(mask)
    
    mri_img_new_v = b.apply_transform(x=mri_img_new_,transform_parameters= {'flip_vertical':True})
    mri_mask_new_v = b.apply_transform(x=mri_mask_new_,transform_parameters = {'flip_vertical':True})
    for img, mask in zip(mri_img_new_v,mri_mask_new_v):
        mri_img_new.append(img)
        mri_mask_new.append(mask)
    
    mri_img_new,mri_mask_new= np.array(mri_img_new), np.array(mri_mask_new)
    print('mri_img_new',mri_img_new.shape)
    print('mri_mask_new',mri_mask_new.shape)
    return mri_img_new,mri_mask_new

def GeneratePatches(mri_img, mri_mask, verbose=False, max_shape=(512,512), window_size = (256, 256), quantile_normalize=False):
    mask_hotshot =[]
    total_mask_patch =[]
    total_mri_patch =[]
    #Determine the window for prostate
    max_height = 0
    max_width = 0
    for slice_mask in mri_mask:
        labels = measure.label(slice_mask>0)
        for lbl in measure.regionprops(labels):
            min_row, min_col, max_row, max_col = lbl.bbox
            height, width = (max_row-min_row), (max_col-min_col)
            if max_height< height:
                max_height = height
            if max_width < width:
                max_width = width
    print('max_height, max_width',max_height, max_width)

     #max_height, max_width #(max_row - min_row), (max_col-min_col)
    print('window_size', window_size)
    mri_img_, mri_mask_ = GetROIOnly(mri_img, mri_mask, window_size)

    #Normalize
    print("Quantile Transformation...")
    '''
    if quantile_normalize:
        depth = mri_img_.shape[0]
        img_long = np.zeros((max_shape[0],max_shape[1]*depth))
        for index, img in enumerate(mri_img_):
            img_long[0:max_shape[0],index*max_shape[1]:(index+1)*max_shape[1]] = img[:,:]
        
        qt = QuantileTransformer(random_state=0)   
        img_long = qt.fit_transform(img_long)
        
        if verbose:
            plt.imshow(img_long)
            plt.show()
        for index in range(depth):
            mri_img_[index,:,:] = img_long[0:max_shape[0],index*max_shape[1]:(index+1)*max_shape[1]]
            if verbose:
                plt.imshow(mri_img_[index])
                plt.show()
    '''
    for slice_mask in mri_mask_:
        mask = np.zeros((slice_mask.shape[0],slice_mask.shape[1],3))
        mask[...,1] = slice_mask==1
        mask[...,0] = slice_mask==0
        mask[...,2] = slice_mask==2
        mask_hotshot.append(mask)
    #1. For original mask, mri
    for mask in mask_hotshot:
        total_mask_patch.append(mask)

    for mri in mri_img_:
        total_mri_patch.append(mri)

    mean_vle = np.mean(np.array(mri_img))

    print('mean_vle', mean_vle)
    #2a. For mask and images of region of interest, global mean
    for (slice_mri, slice_mask) in zip(mri_img_,mri_mask_):
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri)
                plt.show()
        slice_mri_mask = np.array(slice_mask>0, dtype=float) * slice_mri
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()

        slice_mri_mask[slice_mask==0] = mean_vle
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()
        total_mri_patch.append(slice_mri_mask)
  
    for mask in mask_hotshot:
        total_mask_patch.append(mask)

    #2b, for mask and images of ROI, local mean
    for (slice_mri, slice_mask) in zip(mri_img_,mri_mask_):
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri)
                plt.show()
        
        slice_mri_mask = np.array(slice_mask>0, dtype=float) * slice_mri
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()

        slice_mri_mask[slice_mask==0] = np.mean(slice_mri)
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()
        total_mri_patch.append(slice_mri_mask)
  
    for mask in mask_hotshot:
        total_mask_patch.append(mask)

    #3. Negative images with no prostate
    
    for i in range(len(mask_hotshot)):
        mask = np.zeros((slice_mask.shape[0],slice_mask.shape[1],3))
        mask[...,0] = 1.
        total_mask_patch.append(mask)

    for (slice_mri, slice_mask) in zip(mri_img_,mri_mask_):
        
        slice_mri_mask = np.array(slice_mask==0, dtype=float) * slice_mri
        slice_mri_mask[slice_mask>0] = mean_vle
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()
        total_mri_patch.append(slice_mri_mask)
    
    
    #4. Negative image with local mean value:
    '''
    for i in range(len(mask_hotshot)):
        mask = np.zeros((slice_mask.shape[0],slice_mask.shape[1],3))
        mask[...,0] = 1.
        total_mask_patch.append(mask)

    for (slice_mri, slice_mask) in zip(mri_img_,mri_mask_):
        
        slice_mri_mask = np.array(slice_mask==0, dtype=float) * slice_mri
        slice_mri_mask[slice_mask>0] = np.mean(slice_mri)
        if verbose:
            if np.count_nonzero(slice_mask)>0:
                plt.imshow(slice_mri_mask)
                plt.show()
        total_mri_patch.append(slice_mri_mask)
    '''
    return total_mri_patch, total_mask_patch

def generate_train_set(filenames, mask_filenames,max_shape=(512,512), short=False, verbose=False):
    mri_slices = []
    mask_slices = []
    print("Generating datset")
    for filename, mask_filename in zip(filenames,mask_filenames):
        print('filename',filename)
        print('mask_filename',mask_filename)
        scan, mask = slice_mri(filename, mask_filename, max_shape)
        weight_x, weight_y = scan.shape[1], scan.shape[2]
        if weight_x > max_shape[0]:
            max_shape[0] = weight_x
        if weight_y > max_shape[1]:
            max_shape[1] = weight_y

        for slice_data, mask_data in zip(scan, mask):
            if np.count_nonzero(np.isnan(slice_data))> 0.:
                print("NAN")
            if np.count_nonzero(np.isnan(mask_data))> 0.:
                print("NAN")
            slice_data = np.nan_to_num(slice_data)
            mri_slices.append(slice_data)
            mask_slices.append(mask_data)
    if short:
        mask_hotshot = []
        for slice_mask in mask_slices:
            mask = np.zeros((slice_mask.shape[0],slice_mask.shape[1],1))
            mask[...,0] = slice_mask==1
            mask_hotshot.append(mask)
        mri_img, mri_mask = mri_slices, mask_hotshot
        b = ImageDataGenerator()
        mri_img_new_h = b.apply_transform(x=np.array(mri_img),transform_parameters= {'flip_horizontal':True})
        mri_mask_new_h = b.apply_transform(x=np.array(mri_mask),transform_parameters= {'flip_horizontal':True})
        for img, mask in zip(mri_img_new_h,mri_mask_new_h):
            mri_img.append(img)
            mri_mask.append(mask)
        mri_img_new_v = b.apply_transform(x=np.array(mri_img),transform_parameters= {'flip_vertical':True})
        mri_mask_new_v = b.apply_transform(x=np.array(mri_mask),transform_parameters = {'flip_vertical':True})
        for img, mask in zip(mri_img_new_v,mri_mask_new_v):
            mri_img.append(img)
            mri_mask.append(mask)
    else:
        import random
        mri_img, mri_mask = GeneratePatches(mri_slices,mask_slices)

    img_mri = np.array(mri_img)
    mask_array =np.array(mri_mask)
    print('img_mri', img_mri.shape)
    print('mask', mask_array.shape)
    return img_mri, mask_array

###########
#
#   Training/Testing
#
###########


def Run(args):
    #Define the model
    GeneratorModel = model_storage.DualModel(number_of_class=args.nb_class, input_shape=(256, 256))
    model, eval_model = GeneratorModel.CoreModelDual(activation_last="softmax")#.CoreModelStructure(activation_last="softmax")#Inception_V3(final_activation="softmax")#CoreModelStructure(activation_last='sigmoid')#Inception_V3(final_activation="softmax") #CoreModelStructure()#.CoreModelDual()
    #model, eval_model = GeneratorModel.CoreModelStructure(activation_last="softmax")
    print(model.summary())
    from keras.models import model_from_json, load_model
    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON         
    if args.load_model:
        
        print("Proc: Loading the previous model...")
        # load json and create model
        loaded_model_json = None
        with open(args.save_dir + '/trained_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        #Load the weight is available...
    if args.weights is None and args.last_weight:
        file_name_weight = utils.GetLastWeight(args.save_dir)
    else:
        file_name_weight = args.weights
        
    if file_name_weight is not None:
        model.load_weights(file_name_weight)
    else:
        print('No weights are provided. Will test using random initialized weights.')

    #Show the model
    #Testing..
    if not args.testing:
        plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
        train(args, model)  
        file_name_weight = utils.GetLastWeight(args.save_dir)
        #eval_model = model
        if file_name_weight is not None:
            print("Loading the weight...")
            eval_model.load_weights(file_name_weight, by_name=True)
        test(args, eval_model, Onlive=True)
    else:
        if file_name_weight is not None:
            print("Loading the weight...")
            #eval_model.save("./test_single_model.h5")
            #model.save("./train_single_model.h5")
            eval_model.load_weights(file_name_weight, by_name=True)
        #if os.path.exists('./scaler.pkl'):
        #    #scaler = joblib.load('./scaler.pkl')
        #    print("Loaded existing scale transformeer...")
        eval_model.summary()
        test(args, model=eval_model, Onlive=True)#"/home/eminaga/EncryptedData/Challenge/MoNuSeg Training Data/All/")
 
def train(args, model, verbose=False):
    train_data_datagen = ImageDataGenerator()#samplewise_center=True,
                                            #samplewise_std_normalization=True)
                                            #width_shift_range=10,
                                            #shear_range=25,
                                            #rotation_range=45 
                                            #height_shift_range=5, 
                                            #brightness_range=(0.9,1.2),
                                            #zoom_range=[0.9,1.0],
                                            #horizontal_flip=True, 
                                            #vertical_flip=True
                                            
    valid_data_datagen = ImageDataGenerator()#samplewise_center=True,
                                            #samplewise_std_normalization=True)#samplewise_center=True,samplewise_std_normalization=True)#samplewise_center=True)

    seed = 1
    if (args.load_numpy):
        print("loading the previous data...")
        valid_img_dataset = np.load('./img_test_set.npy')#, mmap_mode='r')
        valid_mask_dataset = np.load('./mask_test_set.npy')#, mmap_mode='r')
        train_img_dataset = np.load('./img_train_set.npy')#, mmap_mode='r')
        train_mask_dataset = np.load('./mask_train_set.npy')#, mmap_mode='r')
        print("training set: ",train_img_dataset.shape)
        print("validation set: ",valid_img_dataset.shape)
        print("fitting the data")
        print(np.mean(train_img_dataset[0,:,:,0]))
        print(np.median(train_img_dataset[0,:,:,0]))
        if verbose:
            plt.imshow(train_img_dataset[0,:,:,0])
            plt.show()
            plt.imshow(train_mask_dataset[0,:,:,0])
            plt.show()
        test_img = (train_img_dataset[0,:,:,0]-np.mean(train_img_dataset[0,:,:,0]))/np.std(train_img_dataset[0,:,:,0])
        print(np.mean(test_img))
        print(np.median(test_img))
        selected_randomly_to_fill_the_gap = np.random.choice(train_img_dataset.shape[0], 10)
        if verbose:
            for i in selected_randomly_to_fill_the_gap:
                plt.imshow(train_img_dataset[i,:,:,0])
                plt.show()
                plt.imshow(train_mask_dataset[i])
                plt.show()
        #train_img_dataset -= 0.5
        #valid_img_dataset -= 0.5
        train_data_datagen.fit(train_img_dataset, augment=True, seed=seed)
        train_data_datagen.fit(train_mask_dataset, augment=True, seed=seed)
        valid_data_datagen.fit(valid_img_dataset)
        print(np.mean(train_img_dataset[0,:,:,0]))
        print(np.median(train_img_dataset[0,:,:,0]))

        
        train_input_generator = train_data_datagen.flow(
                train_img_dataset, train_mask_dataset, batch_size=args.batch_size)
        
        valid_input_generator = valid_data_datagen.flow(
                valid_img_dataset, valid_mask_dataset, batch_size=args.batch_size)
    
    # callbacks

    print("Proc: Preprare the callbacks...")
    for index,b in zip([0,1,2,3,4],([1.,10., 2.0],[1.,5., 1.5], [1,2,1], [1,7,2],[10,1,10])):
        print(index, b)
        log = callbacks.CSVLogger(args.save_dir + '/%s_log.csv' %(index))
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/%s_tensorboard-logs' % (index),
                                batch_size=args.batch_size, histogram_freq=args.debug)
    
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_ucent_jaccard_distance', mode="min", factor=args.lr_factor, patience=3, min_lr=args.min_lr,verbose=1)
        #history_register = keras.callbacks.History()

        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_ucent_jaccard_distance', #monitor='val_ucent_jaccard_distance',
                                            save_best_only=True, save_weights_only=True, mode='min',verbose=1)
        print("Done: callbacks are created...")
        #1,5,1.5 Not working well.
        number_o_2 = np.sum((train_mask_dataset[...,2])==1)
        number_o_1 = np.sum((train_mask_dataset[...,1])==1)
        number_o_0 = np.sum((train_mask_dataset[...,0])==1)
        number_o_0_1_2 = number_o_2 + number_o_1 + number_o_0
        number_o_2_r = number_o_2 / number_o_0_1_2
        number_o_1_r = number_o_1 / number_o_0_1_2
        number_o_0_r = number_o_0 / number_o_0_1_2
        print(number_o_0_r,number_o_1_r,number_o_2_r)
        print(number_o_0,number_o_1,number_o_2)
        class_weights = np.array(b)
        class_weights_iou_seg = 2 #number_o_0_r/number_o_1_r
        class_weights_iou_contour = 1.2
        print("Proc: Model loading...")
        DualMode = True
        if DualMode:
            model.compile(optimizer=optimizers.adam(lr=args.lr), #SGD(lr=args.lr, nesterov=True),#
                                #loss=loss_functions.dice_coef,
                                #loss=loss_functions.dice_coef,#loss_functions.weighted_categorical_crossentropy(class_weights),
                                loss=[loss_functions.weighted_categorical_crossentropy(class_weights),
                                    loss_functions.weighted_loss_IOU([class_weights_iou_seg]),
                                    loss_functions.weighted_loss_IOU([class_weights_iou_contour])],
                                #loss=[loss_functions.weighted_categorical_crossentropy(class_weights),# ["categorical_crossentropy"],  # self.iou_loss
                                #loss = loss_functions.weighted_loss_IOU([class_weights_iou_seg]),
                                #loss_functions.weighted_loss_IOU([class_weights_iou_contour])],# ["categorical_crossentropy"]
                                #metrics=[utils.jaccard_distance, "acc", "mse"]
                                metrics={'ucent': [utils.jaccard_distance,utils.dice_coefficient,'acc'], #'acc', self.precision, self.recall, 
                                        'predict_to_seg_01': [utils.jaccard_distance],
                                        'predict_to_contour_01': [utils.jaccard_distance]
                                        },
                                #loss_weights=[1,0.3,1.0]
                                )
        else:
            model.compile(optimizer=optimizers.adagrad(lr=args.lr),
                                #loss=loss_functions.dice_coef,
                                #loss=loss_functions.dice_coef,#loss_functions.weighted_categorical_crossentropy(class_weights),
                                loss=[loss_functions.weighted_categorical_crossentropy(class_weights)],
                                #loss=[loss_functions.weighted_categorical_crossentropy(class_weights),# ["categorical_crossentropy"],  # self.iou_loss
                                #loss = loss_functions.weighted_loss_IOU([class_weights_iou_seg]),
                                #loss_functions.weighted_loss_IOU([class_weights_iou_contour])],# ["categorical_crossentropy"]
                                metrics=[utils.jaccard_distance, "acc", "mse"]
                                )

        train_steps_per_epoch = train_img_dataset.shape[0] // args.batch_size
        valid_steps_per_epoch = valid_img_dataset.shape[0] // args.batch_size
        print("Proc: training...")
        model.fit_generator(generator=utils.image_generator(train_input_generator, DualMode=DualMode, reshape=False),
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=args.epochs,
                                    use_multiprocessing=True,
                                    validation_steps=valid_steps_per_epoch,
                                    validation_data=utils.image_generator(valid_input_generator,DualMode=DualMode, reshape=False),
                                    callbacks=[log, tb, checkpoint,lr_decay, reduce_lr])
        print("Done: training...")
        
def test(args, model, Onlive=True):
    test_img_dataset = np.load('./img_test_set.npy')
    #test_img_dataset -= 0.5# np.mean(test_img_dataset, axis=0)
    valid_data_datagen = ImageDataGenerator()#samplewise_center=True,
                                            #samplewise_std_normalization=True)#, samplewise_std_normalization=True)
    valid_data_datagen.fit(test_img_dataset)
    valid_input_generator = valid_data_datagen.flow(
                test_img_dataset,None, batch_size=args.batch_size)
    valid_steps_per_epoch = test_img_dataset.shape[0] // args.batch_size      
    
    #heatmap_predicted_y_x = model.predict(test_img_dataset),batch_size=args.batch_size)
    for i, img in enumerate(test_img_dataset):
        img_ = img.reshape(1, img.shape[0], img.shape[1],1)
        _,heatmap_predicted_y_x, _ = model.predict(img_)
        print(heatmap_predicted_y_x.shape)
        plt.imshow(heatmap_predicted_y_x[0,:,:,0])
        #plt.imshow(heatmap_predicted_y_x[0,:,:,:])
        plt.show()
        plt.imshow(img_[0,:,:,0])
        plt.show()

    from sklearn.metrics import confusion_matrix

    y_true = np.array([0] * 1000 + [1] * 1000)
    y_pred = heatmap_predicted_y_x > 0.5

    confusion_matrix(y_true, y_pred)

def gp(args):
    #Generate numpy list
    print('path:',args.path)
    onlyfiles = [f for f in listdir(args.path) if (isfile(join(args.path, f)) and (os.path.splitext(f)[1] in [".mhd"]))]
    
    def last_6chars(x):
        return(x[0:6])

    onlyfiles = sorted(onlyfiles, key = last_6chars) 
    segmentation_file_list =[]
    image_file_list =[]
    for file in onlyfiles:
        print(file)
        if ('segmentation' in file):
            segmentation_file_list.append(file)
        else:
            image_file_list.append(file)
    
    from random import shuffle
    x = [i for i in range(len(segmentation_file_list))]
    shuffle(x)
    
    image_file_list_ = []
    for index in x:
        image_file_list_.append(join(args.path, image_file_list[index]))

    image_file_list = image_file_list_

    segmentation_file_list_ = []
    for index in x:
        segmentation_file_list_.append(join(args.path, segmentation_file_list[index]))
    

    segmentation_file_list = segmentation_file_list_

    nr_train = int(round((args.ratio/100)*len(segmentation_file_list)))
    
    img_train_set = image_file_list[0:nr_train]
    mask_train_set = segmentation_file_list[0:nr_train]
    
    img_valid_set = image_file_list[nr_train:len(segmentation_file_list)]
    mask_valid_set = segmentation_file_list[nr_train:len(segmentation_file_list)]
    info_data = {'train': (img_train_set, mask_train_set), 'test' : (img_valid_set,mask_valid_set)}

    for val in info_data.keys():
        data_img, data_mask = info_data[val]
        mri_slices, mask_slices = generate_train_set(data_img,data_mask)
        first_shape = mri_slices.shape
        mri_slices = mri_slices.reshape((first_shape[0], first_shape[1], first_shape[2], 1))
        print(mri_slices.shape)
        #plt.imshow(mri_slices[1,:,:,0])
        #plt.show()
        print("Proc: Save the result...")
        np.save('./img_%s_set.npy' % (val), mri_slices)
        np.save('./mask_%s_set.npy' % (val), mask_slices)
        print('Done: Saving the result...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path where the data are stored",
                    default="./train/")  
    parser.add_argument("--test", help="the path where the data are stored",
                    default="./test/")         
    parser.add_argument("--save_dir", help="store the weights and tensorboard reports",
                    default="./result")
    parser.add_argument("--weights", help="load the weights",
                    default=None)
    parser.add_argument("--ratio", help="ratio of train set",
                    default=80, type=int)
    parser.add_argument("--lr_factor",
                    default=0.5, type=int)
    parser.add_argument("--min_lr",
                    default=0.00000000001, type=float)
    parser.add_argument("--lr",
                    default=0.0001, type=float)
    parser.add_argument("--change_lr_threshold",
                    default=2, type=int)
    parser.add_argument("--nb_class",
                    default=3, type=int)
    parser.add_argument("--batch_size", help="define the batch size, standard 16",
                    default=16, type=int)
    parser.add_argument("--epochs", help="define the batch size, standard 16",
                    default=10, type=int)
    parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
    parser.add_argument('-l', '--load_model', default=False, action='store_true',
                    help="Load a previous model.")
    parser.add_argument('-gp',"--path_generate", help="Run patch generation",
                    action="store_true")
    parser.add_argument('-lw',"--last_weight", help="load the last weight",
                    action="store_true")
    parser.add_argument('-la',"--load_numpy", help="load the numpy data",
                    action="store_true", default=True)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('-t', '--testing', default=False, action='store_true',
                        help="Test the trained model on testing dataset")
    args = parser.parse_args()
    if args.path_generate:
        gp(args)
    else:
        Run(args)