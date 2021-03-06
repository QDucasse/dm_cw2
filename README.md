# Data Mining & Machine Learning: Coursework 2

### Installation

The project was developed through a virtual environment with `virtualenvwrapper`
and we highly recommend to do so as well. However, whether or not you are in a
virtual environment, the installation proceeds as follows:

* For downloading and installing the source code of the project:

  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/dm_cw2
    $ python setup.py install
  ```
* For downloading and installing the source code of the project in a new virtual environment:  

  *Download of the source code & Creation of the virtual environment*
  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/dm_cw2
    $ cd dm_cw1
    $ mkvirtualenv -a . -r requirements.txt VIRTUALENV_NAME
  ```
  *Launch of the environment & installation of the project*
  ```bash
    $ workon VIRTUALENV_NAME
    $ pip install -e .
  ```

Note that wou will need to put the data in the root directory because the files
were too big for GitHub. The base datasets can be found following the link
http://www.macs.hw.ac.uk/~ek19/data/. Another version of the files is also
available if you do not want to perform the transformation yourself but rather
obtain all the created files directly (the files can still be recreated uncommenting
one of the main areas of the `loader.py` file). The datasets used for this second
coursework can be found at: https://www.dropbox.com/s/habmey8caojkawv/data_cw2_nn.zip?dl=0.

The results for the decision trees part can be found under https://www.dropbox.com/s/zoi9743jnr60aeg/dm_cw2_trees.zip?dl=0.

  ---

  ### Objectives and Milestones of the project

  - [X] Data preprocessing
  - [X] Decision Trees cross-validation
  - [X] Decision Trees train-test of different sizes
  - [X] Neural Network overfitting analysis
  - [X] Neural Network cross-validation
  - [X] Neural Network train-test of different sizes
  - [X] Neural Network overfitting analysis
  - [X] Tables answering "Variation in performance with size of the training and testing sets"
  - [X] Tables answering "Variation in performance with change in the learning paradigm"
  - [X] Tables answering "Variation in performance with varying learning parameters in Decision Trees"
  - [X] Tables answering "Variation in performance with varying learning parameters in Neural Network"
  - [X] Tables answering "Variation in performance according to different metrics"  
  - [X] MSc Research Question
  ---

### How to contribute

In order to contribute, first ensure you have the latest version by:

Steps to do once in the beginning:
* Forking the project under your Github profile
* Cloning the project on your computer as explained above
* Setting a remote
```bash
  $ git remote add upstream https://github.com/QDucasse/dm_cw2
```

Steps to do before beginning your work on the project:
* Updating your local repository with the latest changes
```bash
  $ git fetch upstream
  $ git checkout master
  $ git merge upstream/master
```

Steps to do to push your changes:
* Push the changes to your local directory
```bash
  $ git add <files-that-changed>
  $ git commit -m "Commit message"
  $ git push
```
* Open a pull request on `github.com/QDucasse/dm_cw2`
