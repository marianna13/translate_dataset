# Translate HuggingFace Dataset

This script can be used to translate any HuggingFace text dataset in any language to English (also to other languages, see Modifications). It uses [Facebook's m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) model.

## Installation 

Clone this repo:

`git clone https://github.com/marianna13/translate_dataset.git`

And install all the required packages by running this in root directory:

`pip install -r requirements.txt`

## Usage

To use this script run the following command in the terminal:

`python translate_data.py [HuggingFace Dataset Name] [File Name] [start] [end]`

For example, with this command you can translate one file from the [LAION2B-multi](https://huggingface.co/datasets/laion/laion2B-multi) Dataset from index 1000 to index 2000 (1000 samples at all):

`python translate_data.py laion/laion2B-multi part-00000-fc82da14-99c9-4ff6-ab6a-ac853ac82819-c000.snappy.parquet 1000 2000`

## Modification

You can change target language (in the script `tgt_lang="en"`). Also you might wanna change `batch_size` and `num_processes` to make the translation more efficient.
