import cv2
import numpy as np
import os


def batch(Data, Labels, BatchSize = 32):
    """Create a new batch of data
    
    Parameters
    ----------
    Data : numpy.ndarray
        The data to batch
    Labels : numpy.ndarray
        The labels to batch
    BatchSize : int
        The size of the batch to generate

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        An subset of data, and a subset of labels
    """

    # Generate some random indices
    indices = np.random.choice(len(Data), BatchSize, False)
    return Data[indices], Labels[indices]




def loadImage(Path, Color=True):
    """Loads a single image

    Parameters
    ----------
    Path : str
        The full or relative path to the image
    Color : bool
        Whether the image should be loaded in color (True) or grayscale (False)
    
    Returns
    -------
    numpy.ndarray
        The image in a numpy array
    """

    return cv2.imread(Path, Color)



def saveImage(Img, Path):
    """Saves a single image

    Parameters
    ----------
    Img : numpy.ndarray
        The image to save
    Path : str
        The path and name to save the image to
    
    Returns
    -------
    None

    """

    cv2.imwrite(Path, Img)


def loadImageNames(Path):
    """Loads the name of all images in a directory
    Will look for all .jpeg, .jpg, .png, and .tiff files

    Parameters
    ----------
    Path : str
        The full or relative path of the directory

    Returns
    -------
    str, list(str)
        The path, and a list of all the file names

    """
    return sorted( img for img in os.listdir(Path) if (os.path.isfile(os.path.join(Path, img))
                                                    and img.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))))


def loadImagesInDirectory(Path, Color=True, Padding="White",
                          ResizeImages=False, Width=100, Height=100, MaintainAspectRatio=True,
                          Crop=False):
    """Loads all images in a directory
    Will look for all .jpeg, .jpg, .png, and .tiff files
    
    Parameters
    ----------
    Path : str
        The full or relative path of the directory
    Color : bool
        True for loading the image in color, or False for grayscale
    Padding : str
        "White" or "Black"
    ResizeImages : bool
        True to resize the images when loading them
    Width : int
        The width to load the picture at (only if ResizeImages == True)
    Height : int
        The height to load the picture at (only if ResizeImages == True)
    MaintainAspectRatio : bool
        True to maintain aspect ratio (will be padded) (only if ResizeImages == True)
    Crop : bool
        If True, it will fit the shortest side and crop the rest (only if ResizeImages == True)
    
    Returns
    -------
    numpy.ndarray
        An array of all the images of shape [num_images, width, height, channels]
    """

    # Helper functions
    # Handles the different options for resizing images
    def resizeEachImage():
        # If we are to maintain the aspect ratio
        if MaintainAspectRatio == True:

            # If it is cropped we need the shortest edge
            if Crop:
                if loadedImage.shape[0] / float(Height) < loadedImage.shape[1] / float(Width):
                    # Height is further from edge than width
                    resizedImage = resizeImage(loadedImage, Height = Height)
                else:
                    # Width is further from edge than height
                    resizedImage = resizeImage(loadedImage, Width = Width)
            else:
                if loadedImage.shape[0] / float(Height) < loadedImage.shape[1] / float(Width):
                    resizedImage = resizeImage(loadedImage, Width = Width)
                else:
                    resizedImage = resizeImage(loadedImage, Height = Height)
        
        # If not MaintainAspectRatio
        else:
            resizedImage = resizeImage(loadedImage, Height, Width)
        
        #print loadedImage.shape
        #print resizedImage.shape
        return resizedImage



    # Some important variables
    images = []
    imageShapes = []
    maxWidth = 0
    maxHeight = 0

    if Color == True:
        channels = 3
    else:
        channels = 1

    # Get all images in directory
    for img in sorted( img for img in os.listdir(Path) if (os.path.isfile(os.path.join(Path, img))
                                                    and img.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')))):
            # Load the image
            loadedImage = loadImage(os.path.join(Path, img), Color=Color)

            # If we need to resize the images
            if ResizeImages:
                resizedImage = resizeEachImage()
                #print resizedImage.shape
                finalImage = cropOrPad(resizedImage, Height, Width, Padding=Padding)
                #print finalImage.shape
                images.append(finalImage)
            
            # Not resizing
            else:         
                # Collect max width and height
                if loadedImage.shape[1] > maxWidth: maxWidth = loadedImage.shape[1]
                if loadedImage.shape[0] > maxHeight: maxHeight = loadedImage.shape[0]
    
                # Save the image and it's shape
                images.append(loadedImage)
                imageShapes.append(loadedImage.shape)
    
    # If Resize images, set the max height and width
    if ResizeImages:
        maxHeight = Height
        maxWidth = Width

    # Make a numpy array to hold all the images
    # If it is color:
    if Color:
        imageArray = np.empty((len(images), maxHeight, maxWidth, channels), dtype=np.uint8)
    else:
        imageArray = np.empty((len(images), maxHeight, maxWidth), dtype=np.uint8)


    if ResizeImages:
        # Loop through every image and add it to the numpy array
        for i, img in enumerate(images):
            imageArray[i] = img
        
    else:
        # Loop through every image and pad it
        for i, img in enumerate(images):
            img2 = cropOrPad(img, maxHeight, maxWidth, Padding=Padding)
    
            # Insert the image
            imageArray[i] = img2
        
    return imageArray
        




def resizeImage(Img, Height = 0, Width = 0):
    """Resize an image to a desired height and width
    
    Parameters
    ----------
    Img : numpy.ndarray
        The image to resize
    Height : int
        The max height you want the image to be. If 0, it is calculated from Width
    Width : int
        The max width you want the image to be. If 0, it is calculated from Height
    
    Returns
    -------
    numpy.ndarray
        The resized image
    """

    # If both are zero, we don't know what to resize it to!
    if (Height == 0 and Width == 0):
        raise ValueError("Height and Width can't both be 0!")
    elif (Height < 0 or Width < 0):
        raise ValueError("Height or Width can't be below 0")
    elif (Height == 0):
        # We need to caluclate the scale from the width
        scale = float(Width) / Img.shape[1]
    elif (Width == 0):
        # we need to calculate the scale from the height
        scale = float(Height) / Img.shape[0]
    else:
        # In this case, the image will not maintain aspect ratio
        if Img.shape[0] > Height:
            return cv2.resize(Img, (Width, Height), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(Img, (Width, Height), interpolation=cv2.INTER_CUBIC)
    
    # If the scale factor is larger:
    if scale > 1:
        return cv2.resize(Img, None, fx = scale, fy = scale, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(Img, None, fx = scale, fy = scale, interpolation=cv2.INTER_AREA)
    



def cropOrPad(Img, Height, Width, Padding="White"):
    """Puts the image into a new array of Height x Width, and crops or pads as necessary

    Parameters
    ----------
    Img : numpy.ndarray
        The image to put onto the canvas
    Height : int
        The desired height of the returned image
    Width : int
        The desired width of the returned image
    Padding : str
        The color to pad with. "White" or "Black" 
    
    Returns
    -------
    numpy.ndarray
        The image after having been cropped or padded
    """

    startY = int(round((Height - Img.shape[0]) / 2.0))
    startX = int(round((Width - Img.shape[1]) / 2.0))
    endX = 0
    endY = 0

    # If these are less than 0, we must trim some
    imageStartY = 0
    imageStartX = 0
    imageEndY = Img.shape[0]
    imageEndX = Img.shape[1]
    
    if startY < 0:
        imageStartY -= startY
        imageEndY = imageStartY + Height
        startY = 0
        endY = Height
    else:
        endY = startY + Img.shape[0]

    if startX < 0:
        imageStartX -= startX
        imageEndX = imageStartX + Width
        startX = 0
        endX = Width
    else:
        endX = startX + Img.shape[1]
    
    # print "Height: " + str(Height) + ", Width: " + str(Width)
    # print "startY: " + str(startY) + ", startX: " + str(startX)
    # print "endY: " + str(endY) + ", endX: " + str(endX)
    # print "imageStartY: " + str(imageStartY) + ", imageStartX: " + str(imageStartX)
    # print "imageEndY: " + str(imageEndY) + ", imageEndX: " + str(imageEndX)
    # print "Image shape: ", Img.shape
    # exit()

    # If it is a full color image
    if len(Img.shape) == 3:
        if Padding == "White":
            array = np.full((Height, Width, 3), 255, dtype=np.uint8)
        elif Padding == "Black":
            array = np.zeros((Height, Width, 3), dtype=np.uint8)
        else:
            raise ValueError("Unknown parameter for Padding, " + Padding + ". Must be Black or White")
    else:
        if Padding == "White":
            array = np.full((Height, Width), 255, dtype=np.uint8)
        elif Padding == "Black":
            array = np.zeros((Height, Width), dtype=np.uint8)
        else:
            raise ValueError("Unknown parameter for Padding, " + Padding + ". Must be Black or White")
    
    # Insert the image into the array
    array[startY:endY, startX:endX] = Img[imageStartY:imageEndY, imageStartX:imageEndX]
    return array




def displayImage(Img, Name="Image", Batch=False):
    """Displays a single image from an array
    
    Parameters
    ----------
    img : numpy.ndarray
        The image in a numpy array
    name : str
        The name of the window
    Img2 : numpy.ndarray
        A second image to display. If None, will just display one image
    
    Returns
    -------
    None
    """

    if Batch is True:
        for i in Img:
            cv2.imshow(Name, i)
    else:
        cv2.imshow(Name, Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return