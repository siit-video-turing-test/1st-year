"""
Example: transforms.Lambda(lambda x:crop_144.processing(x,[256,288,320,352],28)),

crop_144.processing(x,s1,s2)
s1 : resized size 
s2 : final cropped size
"""


from PIL import Image, ImageOps, ImageEnhance

def resize(img, size, interpolation=Image.BILINEAR):
    output = []
    for i in range(len(size)):
        w, h = img.size
        if (w <= h and w == size[i]) or (h <= w and h == size[i]):
            output.append(img)
        if w < h:
            ow = size[i]
            oh = int(size[i] * h / w)
            output.append(img.resize((ow, oh), interpolation))
        else:
            oh = size[i]
            ow = int(size[i] * w / h)
            output.append(img.resize((ow, oh), interpolation))
                
    return output
    
def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))

    
def center_crop(img, output_size):
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)
        
def three_crop(img):
    output = []
    w, h = img.size
    if w < h:
        top = img.crop((0, 0, w, w))
        center = center_crop(img, (w, w))
        bottom = img.crop((0, h-w, w, h))
        output.append(top)
        output.append(center)
        output.append(bottom)
    else:
        left = img.crop((0, 0, h, h))
        center = center_crop(img, (h, h))
        right = img.crop((w-h, 0, w, h))
        output.append(left)
        output.append(center)
        output.append(right)
    return output

def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)
    
def six_crop(img, size):
    output = []
    w, h = img.size
    tl = img.crop((0, 0, size, size))
    tr = img.crop((w - size, 0, w, size))
    bl = img.crop((0, h - size, size, h))
    br = img.crop((w - size, h - size, w, h))
    center = center_crop(img, (size, size))
    total = img.resize((size, size), Image.BILINEAR)
    output.append(tl)
    output.append(tr)
    output.append(bl)
    output.append(br)
    output.append(center)
    output.append(total)
    return output
    
def twelve_crop(img, size, vertical_flip=False):
    output = []
    for i in range(len(img)):
        first_six = six_crop(img[i], size)
        if vertical_flip:
            imgt = vflip(img[i])
        else:
            imgt = hflip(img[i])
    
        second_six = six_crop(imgt, size)
        t = first_six + second_six
        output.extend(t)
    return output

def processing(img, size1):
    res1 = three_crop(img)
    res2 = twelve_crop(res1, size1)
    return res2

