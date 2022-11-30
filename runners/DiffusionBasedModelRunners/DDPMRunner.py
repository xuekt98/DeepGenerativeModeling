import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.DDPM.DDPMNet import DDPMNet
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm


@Registers.runners.register_with_name('DDPMRunner')
class DDPMRunner(DiffusionBaseRunner):
    def __init__(self, args, config):
        super().__init__(args, config)

    def initialize_model(self, args, config):
        if self.config.use_MP:
            ddpmnet = DDPMNet(config)
        else:
            ddpmnet = DDPMNet(config).to(self.config.device[0])
        ddpmnet.apply(weights_init)
        return ddpmnet

    def initialize_optimizer_scheduler(self, net, args, config):
        optimizer = get_optimizer(config.model.DDPM.optimizer, net.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=config.model.DDPM.scheduler.factor,
                                                               patience=config.model.DDPM.scheduler.patience,
                                                               verbose=True,
                                                               threshold=config.model.DDPM.scheduler.threshold,
                                                               threshold_mode='rel',
                                                               cooldown=config.model.DDPM.scheduler.cooldown,
                                                               min_lr=config.model.DDPM.scheduler.min_lr)
        return [optimizer], [scheduler]

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name) = batch
        x = x.to(self.config.device[0])

        loss = net(x)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))

        grid_size = 4
        sample = net.sample(batch_size=self.config.training.batch_size).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_sample', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_calc(self, net, test_loader, sample_path):
        sample_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        batch_size = self.config.test.batch_size

        for j in range(self.config.test.sample_num):
            sample = net.sample(batch_size)
            save_single_image(sample, sample_path, f'output_{j}.png', to_normal=self.config.data.to_normal)