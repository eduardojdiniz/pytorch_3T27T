#!/usr/bin/env python
# coding=utf-8

import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
from os.path import join as pjoin


__all__: ['HTML']


class HTML:
    """
    This HTML class allows us to save images and write texts into a single HTML
    file. It consists of functions such as <add_header> (add a text header to
    the HTML file), <add_images> (add a row of images to the HTML file), and
    <save> (save the HTML to the disk). It is based on Python library
    'dominate', a Python library for creating and manipulating HTML documents
    using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters
        ----------
        web_dir : str
            A directory that stores the webpage. HTML file will be created at
            <web_dir>/index.html; images will be saved at <web_dir/images/
        title : str
            The webpage name
        refresh : int
            How often the website refresh itself; if 0; no refreshing
        """

        self.title = title
        self.web_dir = web_dir
        self.img_dir = pjoin(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""

        return self.img_dir

    def add_header(self, text):
        """
        Insert a header to the HTML file

        Parameters
        ----------
        text : str
            The header text
        """

        with self.doc:
            h3(text)

    def add_images(self, img_list, txt_list, link_list, width=400):
        """
        Add images to the HTML file

        Parameters
        ----------
        img_list : List[str]
            A list of image paths
        txt_list : List[str]
            A list of image names shown on the website
        link_list : List[str]
            A list of hyperref links; when you click an image, it will redirect
            you to a new page
        """

        # Insert a table
        self.t = table(border=1, style="table-layout: fixed;")
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(img_list, txt_list, link_list):
                    with td(style="word-wrap: break-word;", halign="center",
                            valign="top"):
                        with p():
                            with a(href=pjoin('images', link)):
                                img(style=f"width:{width}px",
                                    src=pjoin('images', im))
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""

        html_file = f"{pjoin(self.web_dir, 'index.html')}"
        with open(html_file, 'wt') as f:
            f.write(self.doc.render())
