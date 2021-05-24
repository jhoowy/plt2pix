import itertools, time, pickle, pprint
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from PIL import Image
from tqdm import tqdm

import utils
from loader.dataloader import get_dataset, get_tag_dict, ColorSpace2RGB
from network import Discriminator
from model.se_resnet import BottleneckX, SEResNeXt
from model.pretrained import se_resnext_half


class tag2pix(object):
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
        self.epoch = args.epoch
        self.batch_size = args.batch_size

        self.gpu_mode = not args.cpu
        self.input_size = args.input_size
        self.color_revert = ColorSpace2RGB(args.color_space)
        self.layers = args.layers
        self.plt_weight = args.plt_weight

        self.load_dump = (args.load is not "")

        self.load_path = Path(args.load)

        self.l1_lambda = args.l1_lambda
        self.guide_beta = args.guide_beta
        self.adv_lambda = args.adv_lambda
        self.save_freq = args.save_freq

        self.two_step_epoch = args.two_step_epoch
        self.brightness_epoch = args.brightness_epoch
        self.save_all_epoch = args.save_all_epoch

        self.palette_num = args.palette_num

        self.start_epoch = 1

        #### load dataset
        if not args.test:
            self.train_data_loader, self.test_data_loader = get_dataset(args)
            self.result_path = Path(args.result_dir) / time.strftime('%y%m%d-%H%M%S', time.localtime())

            if not self.result_path.exists():
                self.result_path.mkdir()

            self.test_images = self.get_test_data(self.test_data_loader, args.test_image_count)
        else:
            # self.test_data_loader = get_dataset(args)
            # self.result_path = Path(args.result_dir)
            raise NotImplementedError('Testing is not implemented for now')


        ##### initialize network
        self.net_opt = {
            'guide': not args.no_guide,
            'relu': args.use_relu,
            'bn': not args.no_bn,
            'cit': args.use_cit,
            'mask': args.use_mask
        }

        if self.net_opt['cit']:
            self.Pretrain_ResNeXT = se_resnext_half(dump_path=args.pretrain_dump, num_classes=cit_class_num, input_channels=1)
        else:
            self.Pretrain_ResNeXT = nn.Sequential()

        self.G = Generator(input_size=args.input_size, layers=args.layers,
                palette_num=self.palette_num, net_opt=self.net_opt)
        
        input_dim = 3 + self.palette_num if args.use_mask else 3
        self.D = Discriminator(input_dim=input_dim, output_dim=1, input_size=self.input_size,
                palette_num=self.palette_num)

        for param in self.Pretrain_ResNeXT.parameters():
            param.requires_grad = False
        if args.test:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.D.parameters():
                param.requires_grad = False

        self.Pretrain_ResNeXT = nn.DataParallel(self.Pretrain_ResNeXT)
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu mode: ", self.gpu_mode)
        print("device: ", self.device)
        print(torch.cuda.device_count(), "GPUS!")

        if self.gpu_mode:
            self.Pretrain_ResNeXT.to(self.device)
            self.G.to(self.device)
            self.D.to(self.device)
            self.BCE_loss.to(self.device)
            self.CE_loss.to(self.device)
            self.L1Loss.to(self.device)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.to(self.device), self.y_fake_.to(self.device)

        if self.load_dump:
            self.load(self.load_path)
            print("continue training!!!!")
        self.end_epoch = self.epoch

        self.print_params()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            print("EPOCH: {}".format(epoch))

            self.G.train()
            epoch_start_time = time.time()

            if epoch == self.brightness_epoch:
                print('changing brightness ...')
                self.train_data_loader.dataset.enhance_brightness(self.input_size)

            max_iter = self.train_data_loader.dataset.__len__() // self.batch_size

            for iter, (original_, sketch_, mask_, palette_) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                if iter >= max_iter:
                    break

                if self.gpu_mode:
                    sketch_, original_, palette_ = sketch_.to(self.device), original_.to(self.device), palette_.to(self.device)

                if self.net_opt['mask']:
                    if self.gpu_mode:
                        mask_ = mask_.to(self.device)
                else:
                    mask_ = None

                # update D network
                self.D_optimizer.zero_grad()

                with torch.no_grad():
                    feature_tensor = self.Pretrain_ResNeXT(sketch_)
                if self.gpu_mode:
                    feature_tensor = feature_tensor.to(self.device)

                D_real, plt_real = self.D(original_, mask_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_f, _ = self.G(sketch_, palette_)
                if self.gpu_mode:
                    G_f = G_f.to(self.device)

                D_f_fake, plt_f_fake = self.D(G_f, mask_)
                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_fake_)

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    plt_real_loss = self.L1Loss(plt_real, palette_)

                    P_real_loss = self.plt_weight * plt_real_loss + self.plt_weight * plt_real_loss

                    plt_f_fake_loss = self.L1Loss(plt_f_fake, palette_)

                    P_f_fake_loss = self.plt_weight * plt_f_fake_loss + self.plt_weight * plt_f_fake_loss
                else:
                    P_real_loss = 0
                    P_f_fake_loss = 0

                D_loss = self.adv_lambda * (D_real_loss + D_f_fake_loss) + (P_real_loss + P_f_fake_loss)

                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_f, G_g = self.G(sketch_, palette_)

                if self.gpu_mode:
                    G_f, G_g = G_f.to(self.device), G_g.to(self.device)

                D_f_fake, plt_f_fake = self.D(G_f, mask_)

                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_real_)

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    plt_f_fake_loss = self.L1Loss(plt_f_fake, palette_)

                    P_f_fake_loss = self.plt_weight * plt_f_fake_loss + self.plt_weight * plt_f_fake_loss
                else:
                    P_f_fake_loss = 0

                L1_D_f_fake_loss = self.L1Loss(G_f, original_)
                L1_D_g_fake_loss = self.L1Loss(G_g, original_) if self.net_opt['guide'] else 0

                G_loss = (D_f_fake_loss + P_f_fake_loss) + \
                         (L1_D_f_fake_loss + L1_D_g_fake_loss * self.guide_beta) * self.l1_lambda

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(
                        epoch, (iter + 1), max_iter, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            with torch.no_grad():
                self.visualize_results(epoch)
                utils.loss_plot(self.train_hist, self.result_path, epoch)

            if epoch >= self.save_all_epoch > 0:
                self.save(epoch)
            elif self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save(epoch)

        print("Training finish!... save training results")

        if self.save_freq == 0 or epoch % self.save_freq != 0:
            if self.save_all_epoch <= 0 or epoch < self.save_all_epoch:
                self.save(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(
            np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))
        

    def test(self):
        self.load_test(self.args.load)

        self.D.eval()
        self.G.eval()

        load_path = self.load_path
        result_path = self.result_path / load_path.stem

        if not result_path.exists():
            result_path.mkdir()

        with torch.no_grad():
            for sketch_, index_, _, cv_tag_ in tqdm(self.test_data_loader, ncols=80):
                if self.gpu_mode:
                    sketch_, cv_tag_ = sketch_.to(self.device), cv_tag_.to(self.device)

                with torch.no_grad():
                    feature_tensor = self.Pretrain_ResNeXT(sketch_)

                if self.gpu_mode:
                    feature_tensor = feature_tensor.to(self.device)

                # D_real, CIT_real, CVT_real = self.D(original_)
                G_f, _ = self.G(sketch_, feature_tensor, cv_tag_)
                G_f = self.color_revert(G_f.cpu())

                for ind, result in zip(index_.cpu().numpy(), G_f):
                    save_path = result_path / f'{ind}.png'
                    if save_path.exists():
                        for i in range(100):
                            save_path = result_path / f'{ind}_{i}.png'
                            if not save_path.exists():
                                break
                    img = Image.fromarray(result)
                    img.save(save_path)

    def visualize_results(self, epoch, fix=True):
        if not self.result_path.exists():
            self.result_path.mkdir()

        self.G.eval()

        # test_data_loader
        original_, sketch_, palette_ = self.test_images
        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        # iv_tag_ to feature tensor 16 * 16 * 256 by pre-reained Sketch.
        with torch.no_grad():
            # feature_tensor = self.Pretrain_ResNeXT(sketch_)
            
            if self.gpu_mode:
                original_, sketch_, palette_= original_.to(self.device), sketch_.to(self.device), palette_.to(self.device)

            G_f, G_g = self.G(sketch_, palette_)

            if self.gpu_mode:
                G_f = G_f.cpu()
                G_g = G_g.cpu()

            G_f = self.color_revert(G_f)
            G_g = self.color_revert(G_g)

        utils.save_images(G_f[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path / 'plt2pix_epoch{:03d}_G_f.png'.format(epoch))
        utils.save_images(G_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path / 'plt2pix_epoch{:03d}_G_g.png'.format(epoch))

    def save(self, save_epoch):
        if not self.result_path.exists():
            self.result_path.mkdir()

        with (self.result_path / 'arguments.txt').open('w') as f:
            f.write(pprint.pformat(self.args.__dict__))
        
        save_dir = self.result_path

        torch.save({
            'G' : self.G.state_dict(),
            'D' : self.D.state_dict(),
            'G_optimizer' : self.G_optimizer.state_dict(),
            'D_optimizer' : self.D_optimizer.state_dict(),
            'finish_epoch' : save_epoch,
            'result_path' : str(save_dir)
            }, str(save_dir / 'plt2pix_{}_epoch.pkl'.format(save_epoch)))

        with (save_dir / 'plt2pix_{}_history.pkl'.format(save_epoch)).open('wb') as f:
            pickle.dump(self.train_hist, f)

        print("============= save success =============")
        print("epoch from {} to {}".format(self.start_epoch, save_epoch))
        print("save result path is {}".format(str(self.result_path)))

    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])

    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        self.start_epoch = checkpoint['finish_epoch'] + 1

        self.finish_epoch = self.args.epoch + self.start_epoch - 1

        print("============= load success =============")
        print("epoch start from {} to {}".format(self.start_epoch, self.finish_epoch))
        print("previous result path is {}".format(checkpoint['result_path']))


    def get_test_data(self, test_data_loader, count):
        test_count = 0
        original_, sketch_, palette_ = [], [], []
        for orig, sket, _, plt in test_data_loader:
            original_.append(orig)
            sketch_.append(sket)
            palette_.append(plt)

            test_count += len(orig)
            if test_count >= count:
                break

        original_ = torch.cat(original_, 0)
        sketch_ = torch.cat(sketch_, 0)
        palette_ = torch.cat(palette_, 0)
        
        self.save_palette(palette_, self.result_path / 'test_image_palettes.png')

        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        if self.gpu_mode:
            original_ = original_.cpu()
        sketch_np = sketch_.data.numpy().transpose(0, 2, 3, 1)
        original_np = self.color_revert(original_)

        utils.save_images(original_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'plt2pix_original.png')
        utils.save_images(sketch_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'plt2pix_sketch.png')

        return original_, sketch_, palette_


    def save_palette(self, plt_tensor, save_file_path):
        plt_image = utils.plt_to_img(plt_tensor)
        # plt_np = self.color_revert(plt_image)
        plt_np = plt_image.data.numpy().transpose(0, 2, 3, 1).astype(np.uint8)

        utils.save_images(plt_np, [plt_np.shape[0], 1], save_file_path)

    def print_params(self):
        params_cnt = [0, 0, 0]
        for param in self.G.parameters():
            params_cnt[0] += param.numel()
        for param in self.D.parameters():
            params_cnt[1] += param.numel()
        for param in self.Pretrain_ResNeXT.parameters():
            params_cnt[2] += param.numel()
        print(f'Parameter #: G - {params_cnt[0]} / D - {params_cnt[1]} / Pretrain - {params_cnt[2]}')
