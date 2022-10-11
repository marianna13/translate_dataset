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
from pathlib import Path
import captionsathome as cah
import torch
import subprocess
import random
from random import randint
import gc
set_verbosity_error()
warnings.filterwarnings("ignore")   

TMP_DATA_DIR = 'DATA1'
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
    torch.cuda.empty_cache()
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt",
                            padding=True).to(device)
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.get_lang_id("en"),
            max_length=30
            ).to(device)
        del encoded
        gc.collect()
        eng_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        del generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        return eng_text


def process_data(d, name, folder, tokenizer, model, device):
    '''
    d:dataset that we want to process
    name:str name of the parquet file (in our case language)
    folder:str folder to save processed data (different folder for each process)
    '''
    torch.cuda.empty_cache()
    d = d.map(lambda example: {'ENG TEXT':translate(example['TEXT'], tokenizer, model, device)},
              batched=True, batch_size=25)
    # d.to_parquet(f'{folder}/{name}.parquet', batch_size=100)
    d.to_parquet(name, batch_size=1000)
    del d
    gc.collect()
    torch.cuda.empty_cache()


def translate_part(tokenizer, model, start, end, small_dataset,device, DATA_DIR):
    '''
    Translates part of the desired dataset.
    tokenizer:transformers.PreTrainedTokenizer
    model:transformers.PreTrainedModel
    start:int start of the range that dataset is splited
    end:int end of the range that dataset is splited
    small_dataset: part of the dataset that is being used
    '''

    part_dataset = small_dataset.select([i for i in range(start, end, 1)])

    name = f'{DATA_DIR}/{start}-{end}.parquet'

    process_data(part_dataset,
                name,
                None,
                tokenizer,
                model,
                device
            )
    torch.cuda.empty_cache()


def worker():
    client = cah.init(
        url="http://cah.io.community/",
        device_id="cluster"
    )
    cache_dir = 'CACHE'

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", use_fast=True, cache_dir=cache_dir)

    # target language
    tokenizer.tgt_lang = "en"

    extra_dir = '/fsx/marianna/translate_dataset/EXTRA'
    files = [f for f in os.listdir(extra_dir) if 'crc' not in f]
    random.shuffle(files)
  

    for file_path in files:



        file_path = f'{extra_dir}/{file_path}'
        client.newJob()
        client.log(client.jobCount())
        client.log(file_path)
        client.log("Processing...")
        

        dataset = load_dataset("parquet", data_files=file_path, cache_dir=cache_dir, split='train')
        file_path = file_path.split('/')[-1]
        num_samples = len(dataset)
        client.log(num_samples)
        n = 50000
        num_proc_per_device = 2
        models = [M2M100ForConditionalGeneration.from_pretrained(
                    "facebook/m2m100_1.2B", cache_dir=cache_dir+'/model').to(f'cuda:{i}') for i in range(num_devices)]*num_proc_per_device

        out = subprocess.check_output("aws s3 ls s3://s-laion/nolang1b-translation/", shell=True)
        done = [str(d.split()[-1]) for d in out.decode("utf-8").split('\n') if d!='']
        done = set(done)
        done = [d for d in done if file_path in d]
        done_idx = [int(d.split('_')[-1].replace('.parquet', '')) for d in done]
            
        for i in range(int(num_samples/n)):
            if i in done_idx:
                continue
            path = f'/{file_path}{i}'
            DATA_DIR = TMP_DATA_DIR + path
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR) 
            parquet_name = f'{file_path}_{i}.parquet'
            out = subprocess.check_output("aws s3 ls s3://s-laion/nolang1b-translation/", shell=True)
            done = [str(d.split()[-1]) for d in out.decode("utf-8").split('\n') if d!='']
            done = set(done)
            if parquet_name in done:
                continue
            if path in os.listdir('DATA1'):
                continue

            DATA_DIR_TRANSLATED = f'{FINAL_DATA_DIR}/{file_path}'

            if not os.path.exists(DATA_DIR_TRANSLATED):
                os.makedirs(DATA_DIR_TRANSLATED)
            
            shard = dataset.shard(
                int(num_samples/n), i, contiguous=True, keep_in_memory=True)

            s = time.time()
            # to make parallel GPU computations possible
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            # number of processes depends on the number of GPUs available, can be varied
            
            num_processes = num_devices*num_proc_per_device

            device_ids = [i for i in range(num_devices)]*num_proc_per_device

            processes = []
            n_samples = len(shard)
            c = int(n_samples/num_processes)
            ranges = [[_, _+c] for _ in range(0, n_samples, c)]
            for rank, rng, device_id, model in zip(range(num_processes), ranges, device_ids, models):
        
                device = torch.device(f"cuda:{device_id}")
                print(f'Using: {device}')
                model = model.half()
                nodel = model.eval()
                model.share_memory()
                start, end = rng
                p = mp.Process( 
                    target=translate_part,
                    args=[tokenizer, model, start, end, shard, device, DATA_DIR]
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            del shard
            gc.collect()
            torch.cuda.empty_cache()


            # combining parquet files for every process into one file

            fs = [pd.read_parquet(DATA_DIR+'/'+path)
                for path in os.listdir(DATA_DIR)]

            translated_parquet = f'{DATA_DIR_TRANSLATED}/{file_path}_{i}.parquet'
            e = time.time()
            print(f'Processed in {round(e-s, 2)} seconds')

            s3_url = f's3://s-laion/nolang1b-translation/{parquet_name}'
            pd.concat(fs).to_parquet(s3_url)

            shutil.rmtree(DATA_DIR)


        client.completeJob()

    return 0


if __name__ == '__main__':
    exit(worker())
    client.bye()
