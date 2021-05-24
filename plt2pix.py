import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from loader.dataloader import ColorSpace2RGB
from torchvision import transforms

# Only for inference
class plt2pix(object):
    def __init__(self, args):
        if args.model == 'tag2pix':
            from network import Generator
        elif args.model == 'senet':
            from model.GD_senet import Generator
        elif args.model == 'resnext':
            from model.GD_resnext import Generator
        elif args.model == 'catconv':
            from model.GD_cat_conv import Generator
        elif args.model == 'catall':
            from model.GD_cat_all import Generator
        elif args.model == 'adain':
            from model.GD_adain import Generator
        elif args.model == 'seadain':
            from model.GD_seadain import Generator
        else:
            raise Exception('invalid model name: {}'.format(args.model))

        self.args = args

        self.gpu_mode = not args.cpu
        self.input_size = args.input_size
        self.color_revert = ColorSpace2RGB(args.color_space)
        self.layers = args.layers

        self.palette_num = args.palette_num

        self.sketch_transform = transforms.Compose([
                                    transforms.Resize((self.input_size, self.input_size), interpolation=Image.LANCZOS),
                                    transforms.ToTensor()])

        ##### initialize network
        self.net_opt = {
            'guide': not args.no_guide,
            'relu': args.use_relu,
            'bn': not args.no_bn,
            'cit': False
        }

        self.G = Generator(input_size=args.input_size, layers=args.layers,
                palette_num=self.palette_num, net_opt=self.net_opt)

        for param in self.G.parameters():
            param.requires_grad = False

        self.G = nn.DataParallel(self.G)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu mode: ", self.gpu_mode)
        print("device: ", self.device)
        print(torch.cuda.device_count(), "GPUS!")

        if self.gpu_mode:
            self.G.to(self.device)
        
        self.G.eval()
        self.load_test(args.load)
        self.G.module.net_opt['guide'] = False

    def colorize(self, input_image, palette=None):
        ''' Colorize input image based on palette

        Parameters:
            input_image (PIL.Image) -- the input sketch image
            palette (np.array)      -- RGB-ordered conditional palette (K x 3).

        Returns:
            G_f (np.array)          -- the colorized result of input sketch image
        '''

        sketch = self.sketch_transform(input_image)
        if palette is None:
            palette = np.zeros(3 * self.palette_num)
            
        palette = torch.FloatTensor((palette / 255.0))
        
        sketch = sketch.reshape(1, *sketch.shape)
        palette = palette.reshape(1, *palette.shape)

        if self.gpu_mode:
            palette = palette.to(self.device)
            sketch = sketch.to(self.device)

        G_f, _ = self.G(sketch, palette)
        G_f = self.color_revert(G_f.cpu())[0]

        return G_f

    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])
