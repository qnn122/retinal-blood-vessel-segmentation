from blood_segmentation.model.dense_unet.net import FCDenseNet57, FCDenseNet67
from blood_segmentation.model.unet.net import Unet
from blood_segmentation.model.laddernet.net import LadderNetv6

import numpy as np
import progressbar

import torch
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def digitize(x,threshold=0.3):
    if x>threshold:
        return min(0.99,2.0*x)
    else:
        return 0.

class RetinalBloodVesselGenerator():
    '''
    Generate retinal blood vessel mask

    '''
    def __init__(self, 
                 net_type='unet', 
                 pretrained_model='./models/unet.pth', 
                 using_gpu=True, 
                 verbose=True):
        '''
        Load trained model
        
        Parameters:
            net_type: can be one of these values `unet`, `dense-unet-4`, `dense-unet-5`, `laddernet`
            pretrained_model: path to weights, don't forget to download the weights before using this class
            using_gpu: whether to use GPU when inferring
            verbose: whether to print progress bar while inferring
         
        '''
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'
        self.net_type = net_type
        self.verbose = verbose

        # Create model object
        if net_type == 'unet':
            self.model = Unet(1).to(self.device)
        elif net_type == 'dense-unet-4':
            self.model = FCDenseNet57(n_classes=2).to(self.device)
        elif net_type == 'dense-unet-5':
            self.model = FCDenseNet67(n_classes=2).to(self.device)
        else:
            self.model = LadderNetv6(inplanes=1).to(self.device)

        # Load pretrained model
        model_checkpoint = torch.load(pretrained_model, map_location=self.device)
        self.model.load_state_dict(model_checkpoint['state_dict'])
        self.patch_size = model_checkpoint.get('patch_size', 256)
        self.max_height = model_checkpoint.get('max_height', -1)

        # Switch model to evaluation mode
        self.model.eval()

        self.convert_to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        if net_type == 'laddernet':
            self.convert_to_tensor = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

        self.digitize = np.vectorize(digitize)

    def create_patches(self, im):
        '''
        Create patches from an image
        '''
        new_size = self.patch_size*int(im.size[0]/self.patch_size),self.patch_size*int(im.size[1]/self.patch_size)
        im = im.resize(new_size)
        image_patches = []
        for i,x in enumerate(range(0,new_size[1],self.patch_size)):
            row_patches= []
            for j,y in enumerate(range(0,new_size[0],self.patch_size)):
                new_im = im.crop((y,x,y+self.patch_size,x+self.patch_size))
                row_patches.append(self.convert_to_tensor(new_im))
            image_patches.append(row_patches)
        return image_patches

    def generate(self, 
                 image, 
                 target_path=None):
        '''
        Generate blood vessel mask
        
        Parameters:
            image: RGB image
            target_path: path to save output image
        
        Returns:
            segmented image if `target_path` is None
        '''
        image_size = image.size
        
        # Scale image
        if self.max_height > 0:
            new_width = int((image.size[0]/ image.size[1]) * self.max_height)
            image = image.resize((new_width, self.max_height))
            
        # Create patches
        patched_image = self.create_patches(image)

        full_image = []
        progress_bar = None
        if self.verbose:
            max_progress_val = sum([len(row) for row in patched_image])
            progress_bar = progressbar.ProgressBar(maxval=max_progress_val, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            progress_bar.start()
        
        # Generate masks for patches
        for i, row in enumerate(patched_image):
            row_im = []
            for j, im in enumerate(row):
                img = im.unsqueeze(0).to(self.device)
                pred = self.model(img)
                if self.net_type == 'laddernet':
                    pred = torch.exp(pred)
                pred = pred.to("cpu").detach().numpy()[0][0]
                row_im.append(pred)
                if progress_bar:
                    progress_bar.update(i*len(row) + j + 1)
            full_image.append(row_im)
        stitch = []

        # Stack patches to build full image
        for i in full_image:
            col_im = []
            for j in i:
                col_im.append(j)
            col_im = np.hstack(col_im)
            stitch.append(col_im)
        stitch = np.vstack(stitch)
        stitch = self.digitize(stitch)
        
        # Store mask to target
        stitch = Image.fromarray((stitch*255).astype(np.uint8))
        stitch = stitch.resize(image_size)
        
        if target_path:
            stitch.save(target_path)
        else:
            return stitch
