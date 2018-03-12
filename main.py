import argparse
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

import preresnet
import preorigin
import preBN
import resnext_101_32x4d

from loader_place import SingleImage
import crop_36

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Place Recognition for Images')
parser.add_argument('image', metavar='DIR', help='input image')
def main():
    global args
    args = parser.parse_args() 

    # (mode, crop, scale)
    ensemble_list = [
       ('resnext-entropy', 36, 232),
       ('resnext-entropy', 36, 240),
       ('resnext-entropy', 36, 248),
       ('resnext-entropy', 36, 256),
       ('resnext-entropy', 36, 264),
       ('preresnet-entropy', 36, 248),
       ('preresnet-entropy', 36, 256),
       ('preresnet-entropy', 36, 264),       
       ('resnext', 36, 232),
       ('resnext', 36, 240),
       ('resnext', 36, 248),
       ('resnext', 36, 256),
       ('resnext', 36, 264),
       ('preresnet', 36, 248),
       ('preresnet', 36, 264),
       ('preresnet', 10, 240),
       ('preresnet', 10, 248),
       ('preBN', 36, 232),
       ('preBN', 36, 240),
       ('preBN', 36, 248),
       ('preBN', 36, 264),
       ('preorigin', 36, 232),
       ('preorigin', 36, 240),
       ('preorigin', 36, 248),
       ('preorigin', 10, 232),
       ('preorigin', 10, 240),

    ]
    
    # Load class index
    idx_to_class = {}
    with open('categories_places365.txt') as f:
        for line in f:
            (val, key) = line.split()
            idx_to_class[int(key)] = val

 

    output_list = []
    prev_net = ''
    for en in ensemble_list:
        if prev_net is not en[0]:
            if en[0]=='resnext':
                model = resnext_101_32x4d.resnext_101_32x4d
                model[10].add_module('1', torch.nn.Linear(2048,365))             
                ckpt_path = 'resnext_c36_best.pth.tar'
            elif en[0]=='preresnet':
                model = preresnet.resnet152(num_classes=365)
                ckpt_path = 'preresnet_best.pth.tar'
            elif en[0]=='preorigin':
                model = preorigin.resnet152(num_classes=365)
                ckpt_path = 'preorigin_best.pth.tar'
            elif en[0]=='preBN':
                model = preBN.resnet152(num_classes=365)
                ckpt_path = 'preBN_best.pth.tar'
            elif en[0]=='preresnet-entropy':
                model = preresnet.resnet152(num_classes=365)
                ckpt_path = 'preresnet_entropy_best.pth.tar'
            elif en[0]=='resnext-entropy':
                model = resnext_101_32x4d.resnext_101_32x4d
                model[10].add_module('1', torch.nn.Linear(2048,365))             
                ckpt_path = 'resnext_entropy_best.pth.tar'
            else:
                raise Exception('Network(%s) is not available'%(en[0]))
        prev_net = en[0]       
 
        model.cuda()
        checkpoint = torch.load(ckpt_path)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        cudnn.benchmark = True

        # Image loading code
        scale = en[2]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if en[1]==10:
            val_loader = torch.utils.data.DataLoader(
                SingleImage(root='./',
                    classlist='categories_places365.txt',
                    img_path=args.image,
                    transform=transforms.Compose([
                        transforms.Resize(scale),
                        transforms.TenCrop(224),
                        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), normalize])(crop) for crop in crops])),
                ])),
                batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False)
        elif en[1]==36:
             val_loader = torch.utils.data.DataLoader(
                SingleImage(root='./',
                    classlist='categories_places365.txt',
                    img_path=args.image,
                    transform=transforms.Compose([
                        transforms.Resize(scale),
                        transforms.Lambda(lambda x:crop_36.processing(x,224)),
                        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), normalize])(crop) for crop in crops])),
                ])),
                batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False)

        else:
            raise Exception('Only 10-crop or 36-crop is available ')
        
        # Evaluate
        output = prediction(val_loader, model)
        output_list.append(output)
        

    # Print
    ensemble_output = torch.cat(output_list,1)
    val5_en, pred5_en = ensemble_output.mean(1).topk(5, 1, True, True)
    
    for idx in range(5):
        sys.stdout.write('(%s :  ' %(idx_to_class[int(pred5_en[0][idx])][3:]))
        sys.stdout.write('%.2f %%) ' %(val5_en[0][idx] * 100.0))
    print(" ")
    

def prediction(val_loader, model):

    predictions = []
    vals = []
    outputs = []
    # switch to evaluate mode
    model.eval()

    num_data = 0
    for i, (input, _) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        bs, ncrops, c, h, w = input_var.size()
        input_var = input_var.view(-1, c, h, w)
               
        # compute output
        output = model(input_var)
        output = nn.functional.log_softmax(output, dim=1)
        output = output.view(bs,ncrops,-1)
        output = torch.exp(output).mean(1,True)
        outputs.append(output)

        # print
        num_data += len(input)

    return torch.cat(outputs,0)



if __name__ == '__main__':
    main()
