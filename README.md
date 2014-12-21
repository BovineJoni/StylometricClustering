Stylometric Clustering
=============
## About
Stylometric Clustering is a prototype of intrinsic plagiarism detection. It automatically extracts stylometric features from a given text and performs a multivariate cluster analysis. The respective clusters represent groups of text passages exhibiting similar stilometric properties and can therefore be associated with the respective number of authors. The input data (text) is represented by articles from the English-language edition of the online encyclopedia Wikipedia.

## Install
##### Python
You need python 2.7 (32-bit):

https://www.python.org/downloads/

_Note: Version 2.7.9 has pip included (which you will need to install the external packages)_

_Note that pip.exe is placed inside your Python installation's Scripts folder (e.g. C:\Python27\Scripts on Windows), which may not be on your path._ [Further instructions on installing pip.](https://pip.pypa.io/en/latest/installing.html)

##### External packages
To install all external dependencies, switch to the project directory and use the following command:

    pip install -r requirements.txt --find-links=./win32_wheelhouse/

For non windows users:

    pip install -r requirements.txt

##### Additional data (too big for this repository)
Textfiles for testing purposes can be found in /project/dir/data.
If you want to save time by using the already extracted features, 
you can download them here (packed 200 MB, extracted about 1 GB):

https://www.dropbox.com/s/utk6d5lrkc6lc6q/extracted_features.zip?dl=0

or

http://web.student.tuwien.ac.at/~e0525640/DA/extracted_features.zip

The zip-archive contains .arff (Weka) and .npy (numpy-arrays) files
for each text-file currently in /project/dir/data in two versions:
RAW-version (full feature set) and PCA95-version (reduced dimensions).

Extract the files to /project/dir/data/extracted_features.
In this case the feature extraction of the respective files will 
skip automatically and the script will proceed with the clustering 
of the loaded data.


## Use
You can use either the console or the gui-version. In both cases you can change
the following properties in the config.cfg file:

    test_file_dir=./data (path to test data)
    auto_split_sentences=1 (turn auto splitting of sentences off or on)
    show_knee_point=1 (turn showing of knee point detection off or on)

_Note: The plotting of the knee point detection is currently only provided for the console version._

##### clusterer.py (console)
Provide the path to a text file (encoding: UTF-8 without BOM) or to a prepared xml test file (placed in /project/dir/data) as argument:
    
    clusterer.py 336_Altruism.txt

If the provided file-path does not exist (e.g. path is not a full path), 
it tries to find the specified file in the data-directory 
(in this case /project/dir/data/336_Autism.txt). After the clustering is done,
the resulting cluster labels are returned. A believe-score between 0 and 1 expresses 
a quality-measure for the calculated number of clusters.

    clusterer.py 336_Altruism.xml
    clusterer.py D:\direct_path_to_some_file.txt

If you provide an xml-file (or if there is an xml-file with the same name and 
in the same directory as a provided txt-file) the resulting clustering will 
get evaluated against the actual author class structure. Once the features of 
a text file are extracted, they are stored in the /extracted_features subdirectory 
of the specified text file-path.

The input text file has to meet the following requirements:
  - encoded in UTF-8 without BOM
  - empty lines indicate paragraph-separation
  - (optional) one sentence per line (set "auto\_split\_sentences" to 1 if sentences are not yet line-separated)

_Note: The auto sentence splitting option gets ignored, if there already 
exists a corresponding xml-file for the provided txt-file.
In that case the author class structure is known and saved as xml.
Splitting or moving of sentences could violate that structure._

##### clusterer_gui.py (gui)
Open the gui version by running the clusterer_gui-script:

    clusterer_gui.py

On top you have three buttons: choose file, save file and process. 
Choose file opens a file dialog where you can select a txt- or xml-file. 
The initial dir is specified by the test\_file\_dir option in config.cfg. 
Open a file and click the process button. The resulting behaviour is 
equivalent to the console version, apart from the graphical output 
(color-coded visualization of cluster labels).

In addition you can use the provided text area to create and process
new text (saving prior to processing is necessary).

### License
Stylometric Clustering, Copyright 2014 Daniel Schneider.
schneider.dnl(at)gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.