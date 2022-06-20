import warnings
from datasets.utils.logging import set_verbosity_error
import time
import sys
import os
import shutil
import pandas as pd
from datasets import load_dataset
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import torch.multiprocessing as mp
import torch
set_verbosity_error()
warnings.filterwarnings("ignore")

DATA_DIR = 'DATA'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if torch.cuda.is_available():
    device = torch.device("cuda:5")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

print(f'Using: {device}')


def translate(example, tokenizer, model):
    '''
    example:Dict[List] with len==batch_size
    tokenizer:transformers.PreTrainedTokenizer
    model:transformers.PreTrainedModel
    returns:
    example with added column ENG TEXT
    '''
    text = example["TEXT"]
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt",
                            padding=True).to(device)
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
        example["ENG TEXT"] = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        return example


def process_data(d, name, folder, tokenizer, model):
    '''
    d:dataset that we want to process
    name:str name of the parquet file (in our case language)
    folder:str folder to save processed data (different folder for each process)
    '''
    d = d.map(lambda example: translate(example, tokenizer, model),
              batched=True, batch_size=10)
    d.to_parquet(f'{folder}/{name}.parquet', batch_size=10)


def translate_part(tokenizer, model, start, end, small_dataset):
    '''
    Translates part of the desired dataset.
    tokenizer:transformers.PreTrainedTokenizer
    model:transformers.PreTrainedModel
    start:int start of the range that dataset is splited
    end:int end of the range that dataset is splited
    small_dataset: part of the dataset that is being used
    '''
    directory = f'{start}-{end}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    part_dataset = small_dataset.select([i for i in range(start, end, 1)])

    # get all languages that are presented in the dataset
    langs = part_dataset.unique('LANGUAGE')
    # go through all the languages

    for lang in langs:
        try:
            tokenizer.src_lang = lang
            process_data(part_dataset.filter(
                lambda example: example["LANGUAGE"] == lang),
                lang,
                directory,
                tokenizer,
                model
            )
        # The model doeasn't support some languages
        except KeyError:
            continue
    # combine all parquet files into one
    fs = [pd.read_parquet(directory+'/'+path)
          for path in os.listdir(directory)]
    pd.concat(fs).to_parquet(f'{DATA_DIR}/{start}-{end}.parquet')
    # remove the directory that is not needed anymore
    shutil.rmtree(directory)


if __name__ == '__main__':
    dataset_name, data_files, start_rng, end_rng = sys.argv[1:]
    start_rng, end_rng = int(start_rng), int(end_rng)

    dataset = load_dataset(dataset_name,
                           data_files=data_files)
    # take desired part of the dataset
    small_dataset = dataset["train"].select(
        [i for i in range(start_rng, end_rng, 1)])
    s = time.time()
    # to make parallel GPU computations possible
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    # number of processes depends on the number of GPUs available, can be varied
    num_processes = 3
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    # target language
    tokenizer.tgt_lang = "en"
    model = M2M100ForConditionalGeneration.from_pretrained(
        "facebook/m2m100_1.2B").to(device)
    model.half()
    # parallel model
    model.share_memory()
    processes = []
    n_samples = end_rng - start_rng
    c = int(n_samples/num_processes)
    ranges = [[_, _+c] for _ in range(0, n_samples, c)]
    for rank, rng in zip(range(num_processes), ranges):
        start, end = rng
        p = mp.Process(target=translate_part, args=[
                       tokenizer, model, start, end, small_dataset])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # combining parquet files for every process in one file
    fs = [pd.read_parquet(DATA_DIR+'/'+path) for path in os.listdir(DATA_DIR)]
    DATA_DIR_TRANSLATED = f'{dataset_name}_{start_rng}_{end_rng}'
    if not os.path.exists(DATA_DIR_TRANSLATED):
        os.makedirs(DATA_DIR_TRANSLATED)
    pd.concat(fs).to_parquet(f'{DATA_DIR_TRANSLATED}.parquet')
    # remove the directories we don't need anymore
    shutil.rmtree(DATA_DIR)
    shutil.rmtree(DATA_DIR_TRANSLATED)

    e = time.time()
    print(f'Processed in {round(e-s, 2)} seconds')
