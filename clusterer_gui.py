#!/usr/bin/env python
# -*- coding: utf-8 -*-

# StylometricClustering, Copyright 2014 Daniel Schneider.
# schneider.dnl(at)gmail.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""StylometricClustering - GUI module
-----------------------------------------------------------

Use:
  Open/save txt- or xml-test-file and click
  process-button to start clustering.

Note:
  txt-file (UTF-8 encoded):
    Each paragraph seperated by an empty line (\\n\\n).
    Each sentence in a seperate line (if auto-split-
    sentences is deactivated).
  xml-testfile (UTF-8 encoded):
    See ".\\336_Altruism.xml" as reference (auto-split-
    sentences does not apply for xml-files). 

Config file ".\\config.cfg":
  test-file-folder:
    path to test-file directory - standard directory
    for open/save-file dialogs.
  auto-split-sentences[0 or 1]:
    0 to deactivate sentence splitting for txt-files,
    1 to activate it.
-----------------------------------------------------------
"""
from __future__ import division
from collections import Counter
from lxml import etree
from ScrolledText import ScrolledText
from tkFileDialog import askopenfilename
from tkFileDialog import asksaveasfilename
import clusterer
import codecs
import ConfigParser
import locale
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import tkMessageBox
import Tkinter as tk
import ttk
print __doc__

class ClustererGui(ttk.Frame):
    """GUI to open/save xml/text-files and visualize clustering."""

    def __init__(self, master=None):
        """Init GUI - get auto-split-sentences-option and standard test-file-folder from config-file."""
        ttk.Frame.__init__(self, master)
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)

        self.createWidgets()
        self.filepath = None
        self.xml_filepath = None
        self.filename = None
        self.article_id = None
        self.extraction = None
        self.author_no = None
        self.correct = None
        self.result = None
        self.colors = []

        config = ConfigParser.ConfigParser()
        config.read("config.cfg")
        params = dict(config.items("params"))
        article_dir = params['test_file_dir']
        self.auto_split_sentences = bool(int(params['auto_split_sentences']))
        self.show_knee_point = bool(int(params['show_knee_point']))
        self.show_knee_point = False # currently not supported in GUI-mode
        self.last_dir = article_dir


    def createWidgets(self):
        """Organize GUI."""
        top=self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        left_frame = ttk.Frame(self, relief="raised", borderwidth=1)
        left_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        left_frame.rowconfigure(0, weight=0)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        buttons_topleft = ttk.Frame(left_frame)
        buttons_topleft.grid(row=0, column=0)

        self.choose_file_btn = ttk.Button(buttons_topleft, text='choose file...',
            command=self.choose_file)
        self.choose_file_btn.grid(row=0, column=0)

        self.save_file_btn = ttk.Button(buttons_topleft, text='save file...',
            command=self.save_file)
        self.save_file_btn.grid(row=0, column=1)
        
        self.extract_feat_btn = ttk.Button(buttons_topleft, text='process',
            command=self.start_featureextr_thread)
        self.extract_feat_btn.grid(row=0, column=2)

        right_frame = ttk.Frame(self)
        right_frame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        self.distr_entry = ScrolledText(right_frame, width=30, height=30)
        self.distr_entry.grid(row=0, column=0, columnspan=2, sticky=tk.N)

        self.test_entry = ScrolledText(right_frame, width=30)
        self.test_entry.grid(row=1, column=0, columnspan=2, sticky=tk.N)

        self.scrolledText = ScrolledText(left_frame, undo=True, wrap=tk.WORD)
        self.scrolledText['font'] = ('Helvetica', '12')
        self.scrolledText.tag_configure('lines', background="#dddddd", foreground="black", font=('Helvetica', 9))
        self.scrolledText.tag_configure('blanks', background="#ffffff", foreground="black", font=('Helvetica', 9))        
        self.scrolledText.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        status_bar = ttk.Frame(self)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        status_bar.columnconfigure(0, weight=1, minsize=100)
        status_bar.columnconfigure(1, weight=1)

        self.status = tk.StringVar()
        self.status.set("ready")
        self.status_label = ttk.Label(status_bar, textvariable=self.status)
        self.status_label.grid(row=0, column=1, padx=10)

        self.progressbar = ttk.Progressbar(status_bar, mode='indeterminate', length=200)
        self.progressbar.grid(row=0, column=0, padx=3)
    
    def choose_file(self):
        """Choose text or xml file dialog."""
        self.filepath = askopenfilename(initialdir=self.last_dir, filetypes=(("text and xml files", ("*.txt","*.xml")),))
        if self.filepath:
            ext = os.path.splitext(self.filepath)[1]

            if ext == ".xml":
                '''save raw-text of xml-file to a new file and print it'''
                self.xml_filepath = self.filepath
                self.filepath = self.create_text_fromXML()

            base = os.path.split(self.filepath)[0]
            self.article_id = os.path.split(base)[1]
            self.filename = os.path.split(self.filepath)[1]
            self.scrolledText.delete(1.0, tk.END)
            self.print_raw_text()
            self.scrolledText.edit_reset()

    def create_text_fromXML(self):
        """Create text-file out of given xml-file."""
        new_filepath = os.path.splitext(self.filepath)[0] + ".txt"
        with codecs.open(self.filepath, 'r', 'UTF-8') as xml_file:
            xml_tree = etree.parse(xml_file)
        
        with codecs.open(new_filepath, 'w', 'UTF-8') as newFile:
            first_entry = True
            for entry in xml_tree.getroot():
                if entry.text is not None:
                    if not first_entry:
                        newFile.write("\n\n")
                    else:
                        first_entry = False
                    newFile.write(entry.text)
        return new_filepath



    def save_file(self):
        """Save text-file-dialog."""
        text = self.scrolledText.get("0.0", tk.END)
        if self.filepath is None:
            name = asksaveasfilename(initialdir=self.last_dir, defaultextension=".txt")
            if name:
                self.filepath = name
            else:
                return
        try:
            with codecs.open(self.filepath, 'w', 'UTF-8') as newFile:
                newFile.write(text.strip())
            self.scrolledText.edit_reset()
            base = os.path.split(self.filepath)[0]
            self.article_id = os.path.split(base)[1]
            self.filename = os.path.split(self.filepath)[1]
            return True
        except Exception:# as e:
            raise


    def start_featureextr_thread(self):
        """Start thread for feature extraction."""
        self.distr_entry.delete(1.0, tk.END)
        self.status.set("processing...")
        if self.filepath is None or self.article_id is None:
            tkMessageBox.showwarning(
                "Save File",
                "Save file for feature extraction.")
            return
        try:
            self.scrolledText.edit_undo()
            self.scrolledText.edit_redo()

            tkMessageBox.showwarning(
                "File changed",
                "File was changed, please save.")
            return
        except tk.TclError:
            self.extraction = clusterer.Clusterer(self.article_id, self.filepath, self.xml_filepath, self.auto_split_sentences, self.show_knee_point)

            self.ftr_extr_thread = threading.Thread(target=self.extract_features)
            self.ftr_extr_thread.daemon = True
            self.progressbar.start()
            self.ftr_extr_thread.start()
            self.after(1000, self.check_feat_thread)

    def check_feat_thread(self):
        """Check if feature extraction thread is still working - if not: visualize cluster-results."""
        if self.ftr_extr_thread.is_alive():
            self.after(1000, self.check_feat_thread)
        else:
            self.status.set("ready")

            # generate author-colormap
            self.colors = [None]*len(set(self.clusters))
            for k in set(self.clusters):  
                temp_color = plt.cm.spectral(np.float(k) / (np.max(self.clusters) + 1))
                if k == 0:
                    temp_color = plt.cm.spectral(0.05)
                self.colors[k] = self.convert_to_hex(temp_color)
            self.configure_colors()

            self.progressbar.stop()
            self.print_author_distr()
            self.print_text()
            if self.correct is not None and self.author_no is not None:
                self.test_entry.delete(1.0, tk.END)
                s = "authors found: {}".format(len(set(self.clusters)))
                s += "\n believe-score: {:.4f}".format(self.believe_score)
                s += "\n\n true number of authors: {}".format(self.author_no)
                s += "\n precision: {:.4f}".format(self.scores[0])
                s += "\n recall: {:.4f}".format(self.scores[1])
                s += "\n f1-score: {:.4f}".format(self.scores[2])
                s += "\n adjusted-rand-index: {:.4f}".format(self.scores[3])
                self.test_entry.insert(tk.INSERT, s)
            else:
                self.test_entry.delete(1.0, tk.END)
                s = "authors found: {}".format(len(set(self.clusters)))
                s += "\n believe-score: {:.4f}".format(self.believe_score)
                self.test_entry.insert(tk.INSERT, s)

    def extract_features(self):
        """Start feature extraction."""
        self.clusters, self.result, self.author_no, self.believe_score, self.scores = self.extraction.calc_cluster()

        if self.result is not None:
            c = Counter(self.result)
            self.correct = c[True] / sum(c.values()) * 100

    def print_text(self):
        """Print raw text with specified author-colors."""
        self.scrolledText.delete(1.0, tk.END)
        f = open(self.filepath)

        line_number = 0
        actual_line_number = 0
        for line in f:
            actual_line_number += 1
            if line.strip():
                s = str(line_number) + ' '+str(self.clusters[line_number]) + ' '+line
                s = line
                line_cluster = str(line_number) + ' '+str(self.clusters[line_number])+ ' '
                line_cluster = ('{:^'+str(14-len(line_cluster))+'}').format(line_cluster)
                self.scrolledText.insert(tk.INSERT, line_cluster, 'lines')
                try:
                    self.scrolledText.insert(tk.INSERT, s, str(self.clusters[line_number]))
                    # if self.result[line_number]:
                    #     # correct assignment - print text foreground in white
                    #     self.scrolledText.insert(tk.INSERT, s, str(self.clusters[line_number]))
                    # else:
                    #     # false assignment - print text foreground in black
                    #     self.scrolledText.insert(tk.INSERT, s, str(self.clusters[line_number]*10**2))
                except IndexError:
                    self.scrolledText.insert(tk.INSERT, s)
                except TypeError:
                        self.scrolledText.insert(tk.INSERT, s, str(self.clusters[line_number]))
                line_number += 1
            else:
                s = line
                self.scrolledText.insert(tk.INSERT, s, 'blanks')
        f.close()

    def print_raw_text(self):
        """Print raw text."""
        f = open(self.filepath)
        for line in f:
            self.scrolledText.insert(tk.INSERT, line)
        f.close()

    def get_distribution(self, l=None):
        """Return Counter with author distribution in percent."""
        if l is None:
            l = self.clusters
        counter = Counter(l)
        sum_counter = sum(counter.values())

        for key in counter.iterkeys():
            counter[key] = counter[key] / sum_counter * 100
        return counter

    def print_author_distr(self):
        """Print author distribution with specified author-colors."""
        self.distr_entry.delete(1.0, tk.END)
        distr = self.get_distribution(self.clusters)

        for index, count in distr.most_common():
            author_i = "author "+str(index)+"{:>20}%\n".format(locale.format(u'%.2f',count))
            self.distr_entry.insert(tk.INSERT, author_i, str(index))

    def convert_to_hex(self, col):
        """Convert inter-tuple to hex-coded string."""
        red = int(col[0]*255)
        green = int(col[1]*255)
        blue = int(col[2]*255)
        return '#{r:02x}{g:02x}{b:02x}'.format(r=red,g=green,b=blue)

    def configure_colors(self):
        """Configure author-specific colors for author-distribution and cluster-results."""
        for i,c in enumerate(self.colors):
            self.scrolledText.tag_configure(str(i), background=c, foreground="white")            
            self.distr_entry.tag_configure(str(i), background=c, foreground="white")
        
if __name__ == '__main__':
    clusterer_gui = ClustererGui()
    clusterer_gui.master.title('Stylometric Clustering')
    clusterer_gui.master.geometry(("{}x{}").format(800,500))
    clusterer_gui.master.minsize(800,500)
    clusterer_gui.mainloop()