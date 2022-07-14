import warnings
from datasets.utils.logging import set_verbosity_error
import time
import sys
import os
import shutil
import pandas as pd
from datasets import load_dataset
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers.onnx import FeaturesManager
import torch.multiprocessing as mp
from pathlib import Path
import captionsathome as cah
from transformers import pipeline
import torch
set_verbosity_error()
warnings.filterwarnings("ignore")   

TMP_DATA_DIR = 'DATA'
FINAL_DATA_DIR = 'DATA_'

if not os.path.exists(TMP_DATA_DIR):
    os.makedirs(TMP_DATA_DIR)

if not os.path.exists(FINAL_DATA_DIR):
    os.makedirs(FINAL_DATA_DIR)

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    torch.cuda.empty_cache()
else:
    assert False, "No GPU available"



def translate(text, tokenizer, model, device):
    '''
    example:Dict[List] with len==batch_size
    tokenizer:transformers.PreTrainedTokenizer
    model:transformers.PreTrainedModel
    returns:
    example with added column ENG TEXT
    '''
    # text = example["TEXT"]
    torch.cuda.empty_cache()
    
    # translator = pipeline(task="translation_xx_to_yy", model=model, tokenizer=tokenizer)
    with torch.no_grad():
        # eng_text = translator(
        #     text, 
        #     return_tensors="pt", 
        #     clean_up_tokenization_spaces=True,
        #     tgt_lang="en"
        #     ) 
        encoded = tokenizer(text, return_tensors="pt",
                            padding=True ).to(device)
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.get_lang_id("en")
            ).to(device)
        eng_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return eng_text


def process_data(d, name, folder, tokenizer, model, device):
    '''
    d:dataset that we want to process
    name:str name of the parquet file (in our case language)
    folder:str folder to save processed data (different folder for each process)
    '''
    d = d.map(lambda example: {'ENG TEXT':translate(example['TEXT'], tokenizer, model, device)},
              batched=True, batch_size=10)
    # d = d.add_column('ENG TEXT', d_eng)
    d.to_parquet(f'{folder}/{name}.parquet', batch_size=100)


def translate_part(tokenizer, model, start, end, small_dataset,device, DATA_DIR):
    '''
    Translates part of the desired dataset.
    tokenizer:transformers.PreTrainedTokenizer
    model:transformers.PreTrainedModel
    start:int start of the range that dataset is splited
    end:int end of the range that dataset is splited
    small_dataset: part of the dataset that is being used
    '''
    directory = DATA_DIR + f'/{start}-{end}'

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
                model,
                device
            )
        # The model doeasn't support some languages
        except KeyError:
            continue
        torch.cuda.empty_cache()
    # combine all parquet files into one
    fs = [pd.read_parquet(directory+'/'+path)
          for path in os.listdir(directory)]
    pd.concat(fs).to_parquet(f'{DATA_DIR}/{start}-{end}.parquet')
    # remove the directory that is not needed anymore
    shutil.rmtree(directory)


def worker():
    client = cah.init(
        url="http://cah.io.community/",
        device_id="cluster"
    )

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", use_fast=True)

    # target language
    tokenizer.tgt_lang = "en"
    
    while client.jobCount() > 0 and not client.shouldDie():
        client.newJob()
        client.log("Processing...")
        client.log(client.tar_url)
        
        url, partition = client.tar_url.split("parquet:")
        url += 'parquet'
        url = url.split('/')[-1]

        print(url)

        DATA_DIR = TMP_DATA_DIR + f'/{url}'

        dataset = load_dataset("laion/laion2B-multi-joined", data_files=url, split="train")
        shard = dataset.shard(
            360, int(partition), contiguous=True, keep_in_memory=True)

        s = time.time()
        # to make parallel GPU computations possible
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # number of processes depends on the number of GPUs available, can be varied
        num_proc_per_device = 3
        num_processes = num_devices*num_proc_per_device

        device_ids = [i for i in range(num_devices)]*num_proc_per_device

        processes = []
        n_samples = len(shard)
        c = int(n_samples/num_processes)
        ranges = [[_, _+c] for _ in range(0, n_samples, c)]
        for rank, rng, device_id in zip(range(num_processes), ranges, device_ids):
            device = torch.device(f"cuda:{device_id}")
            print(f'Using: {device}')
            model = M2M100ForConditionalGeneration.from_pretrained(
                "facebook/m2m100_1.2B").to(device)
            model = model.half()
            nodel = model.eval()
            model.share_memory()
            start, end = rng
            p = mp.Process( 
                target=translate_part,
                args=[tokenizer, model, start, end, dataset, device, DATA_DIR]
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # combining parquet files for every process into one file

        fs = [pd.read_parquet(DATA_DIR+'/'+path)
              for path in os.listdir(DATA_DIR)]

        DATA_DIR_TRANSLATED = f'{FINAL_DATA_DIR}/{url}'

        if not os.path.exists(DATA_DIR_TRANSLATED):
            os.makedirs(DATA_DIR_TRANSLATED)

        pd.concat(fs).to_parquet(
            f'{DATA_DIR_TRANSLATED}/{url}_{partition}.parquet')

        # remove the directories we don't need anymore
        shutil.rmtree(DATA_DIR)

        e = time.time()
        print(f'Processed in {round(e-s, 2)} seconds')

        client.completeJob()

    return 0


if __name__ == '__main__':
    exit(worker())
    client.bye()
