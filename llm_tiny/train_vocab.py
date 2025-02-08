import glob
import json
import os
from tqdm import tqdm
import requests
import sentencepiece as spm
import argparse

DATA_CACHE_DIR = 'F:/datasets/AI-Modelscope/TinyStories/data'

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    # 以写二进制模式打开一个文件以保存下载的内容
    with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

"""
Downloads and unpacks the TinyStories dataset if not already present.
"""
def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories/resolve/master/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")

    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    with open(shard_filenames[0], "r") as f:
        data = json.load(f)

    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

"""
Loads text data from files at the specified path.
"""
def load_text_from_files(path):
    path_list = glob.glob(path)
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data

"""
Yields batches of text data.
    
Parameters:
text_data (list): The list of text data to be batched.
batch_size (int): The size of each batch.
"""
def batch_iterator(text_data, batch_size=648):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i + batch_size]

"""
Trains a tokenizer using SentencePiece with the specified vocabulary size and number of shards.
    
Parameters:
vocab_size (int): The size of the vocabulary.
num_shards (int): The number of shards to use for training.
"""
def train_vocab(vocab_size: int = 32000, num_shards: int = 20):
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece 模型的前缀路径，将用于保存分词器
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1) 将多个分片中的文本导出为单个文本文件 tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 使用 SentencePiece 训练分词器
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,  # 输入文件为之前生成的 tiny.txt
        model_prefix=prefix,  # 模型前缀路径
        model_type="bpe",  # 使用 Byte-Pair Encoding (BPE) 训练分词器
        vocab_size=vocab_size,  # 词汇表大小
        self_test_sample_size=0,  # 自测样本大小设置为 0
        input_format="text",  # 输入文件格式为纯文本
        character_coverage=1.0,  # 覆盖所有字符（包括非常见字符）
        num_threads=os.cpu_count(),  # 使用 CPU 的线程数
        split_digits=True,  # 拆分数字
        allow_whitespace_only_pieces=True,  # 允许仅由空格组成的词元
        byte_fallback=True,  # 启用字节级回退
        unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
        normalization_rule_name="identity"  # 使用“identity”归一化规则
    )

    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=bool, default=True, help="download the dataset")
    parser.add_argument("--vocab_size", type=int, default=4096, help="vocab size")
    args = parser.parse_args()
    if args.download:
        download()
    train_vocab(args.vocab_size)