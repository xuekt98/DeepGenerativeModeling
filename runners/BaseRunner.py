import pdb

import yaml
import os
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file


class BaseRunner(ABC):
    def __init__(self, args, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.args = args  # args from command line
        self.config = config  # config from configuration file

        # set training params
        self.global_epoch = 0  # global epoch
        if args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.args.image_path, \
        self.args.model_path, \
        self.args.log_path, \
        self.args.sample_path, \
        self.args.sample_to_calc_path = make_save_dirs(self.args,
                                                       prefix=self.config.model.name,
                                                       suffix=self.config.data.dataset)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.args.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.args,
                                                                                             self.config,
                                                                                             is_test=self.args.test)

        # initialize EMA
        self.use_ema = False if not self.config.__contains__('EMA') else self.config.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.EMA.ema_decay)
            self.update_ema_interval = self.config.EMA.update_ema_interval
            self.start_ema_step = self.config.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.use_DDP:
            if self.config.use_MP:  # simultaneously using ModelParallel and DistributedDatsParallel
                self.net = DDP(self.net)
            else:
                self.net = DDP(self.net, device_ids=[self.config.local_rank], output_device=self.config.local_rank)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.args.model_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, args, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(args, config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, args, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            print(f"load model {self.config.model.name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if not self.args.test:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.use_DDP:
                self.ema.restore(self.net.module)
            else:
                self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.args.image_path, str(self.global_step)))
        if self.config.use_DDP:
            self.sample(self.net.module, train_batch, sample_path, stage='train')
            # self.sample(self.net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.net, train_batch, sample_path, stage='train')
            # self.sample(self.net, val_batch, sample_path, stage='val')
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def initialize_model(self, args, config):
        """
        initialize model
        :param args: args
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, args, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param args: agrs
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_calc(self, net, test_loader, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        print(self.__class__.__name__)

        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_sampler = None
        val_sampler = None
        if self.config.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.training.batch_size,
                                      shuffle=False,
                                      num_workers=self.config.training.num_workers,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.training.batch_size,
                                    shuffle=False,
                                    num_workers=self.config.training.num_workers,
                                    drop_last=True,
                                    sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.training.batch_size,
                                      shuffle=True,
                                      num_workers=self.config.training.num_workers,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.training.batch_size,
                                    shuffle=True,
                                    num_workers=self.config.training.num_workers,
                                    drop_last=True)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        print(
            f"start training {self.config.model.name} on {self.config.data.dataset}, {len(train_loader)} iters per epoch")

        try:
            for epoch in range(start_epoch, self.config.training.n_epochs):

                if self.config.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)

                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01)
                self.global_epoch = epoch
                for train_batch in pbar:
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        # pdb.set_trace()
                        loss = self.loss_fn(net=self.net,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_step,
                                            opt_idx=i,
                                            stage='train')
                        self.optimizer[i].zero_grad()
                        loss.backward()
                        self.optimizer[i].step()

                        losses.append(loss.detach().mean())
                        if self.scheduler is not None:
                            self.scheduler[i].step(loss)

                    if self.use_ema and self.global_step % self.update_ema_interval == 0:
                        self.step_ema()

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )

                    with torch.no_grad():
                        # if self.global_step % 10 == 0:
                        #     val_batch = next(iter(val_loader))
                        #     self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                        if self.global_step % int(self.config.training.sample_interval * epoch_length) == 0:
                            if not self.config.use_DDP or (self.config.use_DDP and self.config.local_rank) == 0:
                                val_batch = next(iter(val_loader))
                                self.sample_step(val_batch=val_batch, train_batch=train_batch)
                                torch.cuda.empty_cache()

                # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    if not self.config.use_DDP or (self.config.use_DDP and self.config.local_rank) == 0:
                        with torch.no_grad():
                            print("validating epoch...")
                            average_loss = self.validation_epoch(val_loader, epoch)
                            torch.cuda.empty_cache()
                            print("validating epoch success")

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    if not self.config.use_DDP or (self.config.use_DDP and self.config.local_rank) == 0:
                        with torch.no_grad():
                            print("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.args.model_path, f'exception_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.args.model_path, f'exception_optim_sche_{temp}.pth'))
                                remove_file(os.path.join(self.args.model_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.args.model_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.args.model_path,
                                                    f'latest_model_{epoch + 1}.pth'),
                                       _use_new_zipfile_serialization=False)
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.args.model_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'),
                                       _use_new_zipfile_serialization=False)

                            # save top_k checkpoints
                            model_ckpt_name = os.path.join(self.args.model_path,
                                                           f'model_checkpoint_{average_loss:.2f}_epoch={epoch + 1}.pth')
                            optim_sche_ckpt_name = os.path.join(self.args.model_path,
                                                                f'checkpoint_{average_loss:.2f}_epoch={epoch + 1}.pth')

                            save_flag = True
                            remove_flag = True
                            for i in range(self.args.save_top):
                                top_key = f'top_{i}'
                                if top_key not in self.topk_checkpoints:
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    if save_flag:
                                        print(f"saving top_{i} checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                        torch.save(model_states,
                                                   model_ckpt_name,
                                                   _use_new_zipfile_serialization=False)
                                        torch.save(optimizer_scheduler_states,
                                                   optim_sche_ckpt_name,
                                                   _use_new_zipfile_serialization=False)
                                        save_flag = False
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:

                                        print("remove " + self.topk_checkpoints[top_key]["ckpt_name"])
                                        print(
                                            f"saving top_{i} checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                        temp_average_loss = self.topk_checkpoints[top_key]['loss']
                                        temp_model_ckpt_name = self.topk_checkpoints[top_key]['model_ckpt_name']
                                        temp_optim_sche_ckpt_name = self.topk_checkpoints[top_key]['optim_sche_ckpt_name']

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        if save_flag:
                                            torch.save(model_states,
                                                       model_ckpt_name,
                                                       _use_new_zipfile_serialization=False)
                                            torch.save(optimizer_scheduler_states,
                                                       optim_sche_ckpt_name,
                                                       _use_new_zipfile_serialization=False)
                                            save_flag = False

                                        average_loss = temp_average_loss
                                        model_ckpt_name = temp_model_ckpt_name
                                        optim_sche_ckpt_name = temp_optim_sche_ckpt_name
                            if remove_flag:
                                remove_file(model_ckpt_name)
                                remove_file(optim_sche_ckpt_name)
        except BaseException as e:
            if not self.config.use_DDP or (self.config.use_DDP and self.config.local_rank) == 0:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                temp = 0
                while temp < self.global_epoch + 1:
                    remove_file(os.path.join(self.args.model_path, f'exception_model_{temp}.pth'))
                    remove_file(
                        os.path.join(self.args.model_path, f'exception_optim_sche_{temp}.pth'))
                    temp += 1
                torch.save(model_states,
                           os.path.join(self.args.model_path, f'exception_model_{self.global_epoch+1}.pth'),
                           _use_new_zipfile_serialization=False)
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.args.model_path, f'exception_optim_sche_{self.global_epoch+1}.pth'),
                           _use_new_zipfile_serialization=False)

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        if test_dataset is None:
            test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.args.sample_to_calc:
            sample_path = self.args.sample_to_calc_path
            if self.config.use_DDP:
                self.sample_to_calc(self.net.module, test_loader, sample_path)
            else:
                self.sample_to_calc(self.net, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            test_batch = next(test_iter) if self.config.test.has_condition else None
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                sample_path = os.path.join(self.args.sample_path, str(i))
                if self.config.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')
