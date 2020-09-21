import cv2
import glob

sun_data_path = '/scr1/workspace/dataset/sun/SUN2012/Images'

def resize(filename='bg3.png',output_name=None):
    W = 256.
    oriimg = cv2.imread(filename)
    height, width, depth = oriimg.shape
    imgScale = W/width
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    newimg = cv2.resize(oriimg,(int(newX),int(newX)))
    if output_name is not None:
        cv2.imwrite(output_name,newimg)
    else:
        cv2.imwrite (filename, newimg)

if __name__ == '__main__':
    cut_ratio = 21
    for i,file in enumerate(glob.glob(sun_data_path+'/*/*/*.jpg')):
        if i%cut_ratio==1:
            output = './sun_textures/{}.jpg'.format (int(i/cut_ratio))
            resize(file,output)
        print(i)