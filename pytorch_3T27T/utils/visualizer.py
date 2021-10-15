#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
from os.path import join as pjoin
import sys
import ntpath
import time
import visdom
from .helpers import torch2numpy, save_image, mkdirs
from .html import HTML
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """
    This function will save images stored in 'visuals' to the HTML file
    specified by 'webpage'.

    Parameters
    ----------
    webpage : HTML
        The HTML webpage class that stores these images (see html.py)
    visuals : OrderedDict
        An ordered dictionary that stores (name, images) pairs.
        Images are either torch.Tensor or numpy.Array
    image_path : str
        The string is used to create image paths
    aspect_ratio : float
        The aspect ratio of saved images
    width : int
        The images will be resized to width x width
    """

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    img_list, txt_list, link_list = [], [], []

    for label, im_data in visuals.items():
        im = torch2numpy(im_data)
        image_name = f'{name}_{label}.png'
        save_path = pjoin(image_dir, image_name)
        os.makedirs(pjoin(image_dir, label), exist_ok=True)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
        img_list.append(image_name)
        txt_list.append(label)
        link_list.append(image_name)
    webpage.add_images(img_list, txt_list, link_list, width=width)


class Visualizer():
    """
    This class includes several functions that can display/save images and
    print/save logging information. It uses a Python library 'visdom' for
    display, and a Python library 'dominate' (wrapped in 'HTML') for creating
    HTML files with images.
    """

    def __init__(self, config):
        """
        Initialize the Visualizer class

        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object that saves HTML filters
        Step 4: create a logging file to store training losses

        Parameters
        ----------
        config : BaseOptions
            Stores all the experiment flags
        """

        self.config = config  # cache the option
        if config.display_id is None:
            # Just a random display ID
            self.display_id = np.random.randint(100000) * 10
        else:
            self.display_id = config.display_id
        self.use_html = config.isTrain and not config.no_html
        self.win_size = config.display_winsize
        self.name = config.name
        self.port = config.display_port
        self.saved = False

        # Connect to a visdom server given <display_port> and <display_server>
        if self.display_id > 0:
            self.plot_data = {}
            self.ncols = config.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=config.display_server,
                                         port=config.display_port,
                                         env=config.display_env)
            else:
                self.vis = visdom.Visdom(
                    port=2004,
                    base_url=os.environ['tensorboard_base_url'] + '/visdom'
                )

            if not self.vis.check_connection():
                self.create_visdom_connections()

        # Create an HTML object at <ckpt_dir>/web/; images will be saved
        # under <ckpt_dir>/web/images/
        if self.use_html:
            self.web_dir = pjoin(config.ckpt_dir, config.name, 'web')
            self.img_dir = pjoin(self.web_dir, 'images')
            print(f'create web directory {self.web_dir}...')
            mkdirs([self.web_dir, self.img_dir])

        # Create a logging file to store training losses
        self.log_name = pjoin(config.ckpt_dir, config.name, 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'=========== Training Loss ({now}) ============\n')

    def reset(self):
        """Reset the self.saved status"""

        self.saved = False

    def create_visdom_connections(self):
        """
        If the program could not connect to Visdom server, this function will
        start a new server at port <self.port>
        """

        cmd = f'{sys.executable} -m visdom.server -p {self.port} &>/dev/null &'
        print('\n\nCould not connect to Visdom server.\n')
        print('Trying to start a server....')
        print(f'Command: {cmd}')
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """
        Display current results on visdom; save current results to an HTML file

        Parameters
        ----------
        visuals : OrderedDict
            Dictionary of images to display or save
        epoch : int
            The current epoch
        save_result : bool
            If true, save the current results to an HTML file
        """

        # Show images in the browser using visdom
        if self.display_id > 0:
            ncols = self.ncols
            # Show all the images in one visdom panel
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                # Create a table css
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px;
                        white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px;
                        outline: 4px solid black}
                        </style>""" % (w, h)
                # Create a table of images
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    img_numpy = torch2numpy(image)
                    label_html_row += f'<td>{label}</td>'
                    images.append(img_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += f'<tr>{label_html_row}</tr>'
                        label_html_row = ''
                white_img = np.ones_like(img_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_img)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += f'<tr>{label_html_row}</tr>'
                try:
                    self.vis.images(images, ncols, 2, self.display_id + 1,
                                    None, dict(title=f'{title} images'))
                    label_html = '<table>{label_html}</table>'
                    self.vis.text(table_css + label_html,
                                  win=self.display_id + 2,
                                  opts=dict(title=f'{title} labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()
            # Show each image in a separate visdom panel
            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        img_numpy = torch2numpy(image)
                        self.vis.image(img_numpy.transpose([2, 0, 1]),
                                       self.display_id + idx, None,
                                       dict(title=label))
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        # Save images to an HTML file if they haven't been saved.
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            # Save images to the disk
            for label, image in visuals.items():
                img_numpy = torch2numpy(image)
                img_path = pjoin(self.img_dir, f'epoch{epoch:.3d}_{label}.png')
                save_image(img_numpy, img_path)
            # Update website
            webpage = HTML(self.web_dir, f'Experiment name = {self.name}',
                           refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header(f'epoch [{n}]')
                img_list, txt_list, link_list = [], [], []

                for label, img_numpy in visuals.items():
                    img_numpy = torch2numpy(image)
                    img_path = f'epoch{n:.3d}_{label}.png'
                    img_list.append(img_path)
                    txt_list.append(label)
                    link_list.append(img_path)
                webpage.add_images(img_list, txt_list, link_list,
                                   width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """
        Display the current losses on visdom display: dictionary of error
        labels and values

        Parameters
        ----------
        epoch : int
            Current epoch
        counter_ratio : float
            Pogress (percentage) in the current epoch, between 0 to 1
        losses : OrderedDict
            Training losses stored in the format of (name, float) pairs
        """

        if len(losses) == 0:
            return

        plot_name = '_'.join(list(losses.keys()))

        if plot_name not in self.plot_data:
            self.plot_data[plot_name] = {'X': [], 'Y': [], 'legend':
                                         list(losses.keys())}

        plot_data = self.plot_data[plot_name]
        plot_id = list(self.plot_data.keys()).index(plot_name)

        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([losses[k] for k in plot_data['legend']])

        try:
            n_losses = len(plot_data['legend'])
            self.vis.line(
                X=np.stack([np.array(plot_data['X'])] * n_losses, 1),
                Y=np.array(plot_data['Y']),
                opts={'title': self.name + ' loss over time',
                      'legend': plot_data['legend'], 'xlabel': 'epoch',
                      'ylabel': 'loss'},
                win=self.display_id - plot_id
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        Print current losses on console; also save the losses to the disk

        Parameters
        ----------
        epoch : int
            Current epoch
        iters : int
            Current training iteration during this epoch.
            Reset to 0 at the end of every epoch
        losses : OrderedDict
            Training losses stored in the format of (name, float) pairs
        t_comp : float
            Computational time per data point (normalized by batch_size)
        t_data : float
            Data loading time per data point (normalized by batch_size)
        """

        msg = f'(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, '
        msg += f'data: {t_data:.3f}) '
        for k, v in losses.items():
            msg += f'{k}: {v:.3f} '
        print(msg)

        with open(self.log_name, "a") as log_file:
            log_file.write(f'{msg}\n')
