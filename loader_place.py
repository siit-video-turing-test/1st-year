from torchvision.datasets.folder import *

def find_classes_list(dir, fn):
    '''
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    '''
    classes = []
    class_to_idx = {}
    listfn = os.path.join(dir,fn)
    for line in open(listfn,'r'):
        cl, idx = line.split()
        classes.append(cl)
        class_to_idx[cl]=idx
        #print(path, idx)
    return classes, class_to_idx


def make_dataset_list(dir, fn, root):
    images = []
    listfn = os.path.join(dir,fn)
    for line in open(listfn,'r'):
        subpath, idx = line.split()
        if subpath[0] == '/':
            path = os.path.join(root, subpath[1:])
        else:
            path = os.path.join(root, subpath) 
        item = (path, int(idx))
        images.append(item)
    
    #images = images[:2560]
    return images

class ImageFolderFromList(ImageFolder):
    
    def __init__(self, root, classlist, datalist, datadir=None, listdir=None, transform=None, target_transform=None,
                 loader=default_loader):
        if listdir is not None:
            listroot = os.path.join(root,listdir)
        else:
            listroot = root
        if datadir is not None:
            dataroot = os.path.join(root,datadir)
        else:
            dataroot = root

        classes, class_to_idx = find_classes_list(listroot, classlist)        
        imgs = make_dataset_list(listroot, datalist, dataroot)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    #def __getitem__(self, idx):
    #    return super(ImageFolderFromList, self).__getitem__(index)
 
def image_path(path):
    images = []
 
    item = (path, 0)
    images.append(item)
    
    #images = images[:2560]
    return images

def find_classes_list(fn):
    '''
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    '''
    classes = []
    class_to_idx = {}
    listfn = fn
    for line in open(listfn,'r'):
        cl, idx = line.split()
        classes.append(cl)
        class_to_idx[cl]=idx
        #print(path, idx)
    return classes, class_to_idx
       
class SingleImage(ImageFolder):
    
    def __init__(self, root, classlist, img_path, transform=None, target_transform=None,
                 loader=default_loader):
        '''
        if listdir is not None:
            listroot = os.path.join(root,listdir)
        else:
            listroot = root
        if datadir is not None:
            dataroot = os.path.join(root,datadir)
        else:
            dataroot = root
        '''
        classes, class_to_idx = find_classes_list(classlist)        
        imgs = image_path(img_path)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

# Test
if __name__ == '__main__':

    data = ImageFolderFromList('/home/sharedfolder/Places', classlist='categories_places365.txt', datalist='places365_train_challenge.txt' , datadir='challenge/train', listdir='devkit')
    data1 = ImageFolder('/home/sharedfolder/Places/challenge/train/a')

    img, target = data.__getitem__(0)
    img1, target1 = data1.__getitem__(0)

    print(img, target)
    print(img1, target1)

    

