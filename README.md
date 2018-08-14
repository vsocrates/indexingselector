# indexingselector
Models to select articles that need to be indexed in MEDLINE

This project was developed as a part of the LHNCBC Summer Biomedical Informatics Training Program. It includes a number of models and scripts to train and test the automated selective indexing of articles from PubMed to be indexed by MEDLINE. 

## Installation

```
git clone https://github.com/vsocrates/indexingselector.git
cd indexingselector
python network_runner.py [whole bunch of params, see program_tags.txt]
```
Please check the [Wiki](https://github.com/vsocrates/indexingselector/wiki), which has way more information on the various parameters that can be used and what all the scripts do.

## Dependencies

nltk: http://www.nltk.org/install.html  
gensim: https://radimrehurek.com/gensim/install.html  
lxml: https://lxml.de/installation.html  
sklearn: http://scikit-learn.org/stable/install.html  
Keras (**version 2.1.6**):  https://keras.io/#installation   
* There is a slight caveat here. There was some issues with stateful metrics so we have to go in and change the code. 
* In your `site-packages` folder, find the `keras/callbacks.py` file (e.g. `/slurm_storage/socratesv2/anaconda3/envs/tensorflow14/lib/python3.5/site-packages/keras/callbacks.py`) and change **line 871** from `summary_value.simple_value = value.item()` to `summary_value.simple_value = value`. This error is fixed in later versions of the code, but compatibility hasn't been tested yet.   

Tensorflow (**version 1.4.1**): https://www.tensorflow.org/install/   
* Make sure to install 1.4.1 (or 1.4.0 on Windows). Anything later is only prebuilt for CUDA 9, which the LHNCBC DGX-1 does not support as of this commit    
