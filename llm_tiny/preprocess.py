import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = 'F:/datasets/AI-Modelscope/TinyStories/data'
TOKENIZER_MODEL = "F:/datasets/AI-Modelscope/TinyStories/data/tok4096.model"

"""
Process a data shard by tokenizing its text and saving the tokens as a binary file.

Parameters:
args: tuple, containing shard ID and shard filename.
vocab_size: int, size of the vocabulary, used to determine the output file path.
tokenizer_model_path: str, path to the tokenizer model.
"""
def process_shard(args, vocab_size, tokenizer_model_path):

    shard_id, shard = args
    enc = Tokenizer(tokenizer_model_path)

    with open(shard, "r") as f:
        data = json.load(f)

    all_tokens = []

    # 遍历每一个例子，tqdm显示进度条
    for example in tqdm(data, position=shard_id):
        # 提取故事文本，并去除首尾空白字符
        text = example["story"]
        text = text.strip()  # 去掉首尾空白字符

        # 对文本进行编码，使用BOS（开始标志）但不使用EOS（结束标志）
        tokens = enc.encode(text, bos=True, eos=False)
        # 将当前文本的token添加到总token列表
        all_tokens.extend(tokens)

    # 将所有的token转换为uint16类型的NumPy数组
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    if vocab_size == 0:
        # 如果词汇表大小为0，使用默认的Llama 2分词器，将文件保存到原路径
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    # Save the tokens as a binary file
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


"""
Preprocess all data shards and save the tokenized data as binary files.

Parameters:
vocab_size: int, size of the vocabulary, used to determine the output file path.
"""
def pretokenize(vocab_size):

    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    fun = partial(process_shard, vocab_size=vocab_size, tokenizer_model_path=TOKENIZER_MODEL)

    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")

"""
Load pre-tokenized data from disk and return it as PyTorch tensors!!!
"""
class PretokDataset(torch.utils.data.IterableDataset):
    """
    Initialize the dataset.

    Parameters:
    split: str, the dataset split ('train' or 'test').
    max_seq_len: int, the maximum sequence length for generating input-output sequences.
    vocab_size: int, the size of the vocabulary.
    vocab_source: str, the source of the vocabulary ('llama2' or 'custom').
    """
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split  # 数据集划分（训练集或测试集）
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.vocab_size = vocab_size  # 词汇表大小
        self.vocab_source = vocab_source  # 词汇表来源

    """
    Return an iterator that loads data in batches and generates model input/output.
    """
    def __iter__(self):
        # 获取DataLoader的worker信息（用于并行数据加载）
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0  # worker ID
        # 获取分布式训练的rank信息（用于多GPU训练）
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 基于worker_id和rank生成唯一的随机数种子，确保数据在每个worker和rank之间是唯一的
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        # 根据词汇表来源决定数据路径
        if self.vocab_source == "llama2":
            # 如果使用 Llama 2 词汇表，.bin 文件和 .json 文件在同一目录下
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # 如果使用自定义词汇表，.bin 文件在 tok{N} 目录下
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # 根据数据集划分使用不同的分片文件
        # 训练集使用所有分片文件，测试集只使用第一个分片
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"在 {bin_dir} 中未找到任何 .bin 文件"

        while True:
            # 随机打乱分片文件
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # 使用 memmap 读取文件，使得数据留在磁盘上，减少内存占用
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                # 计算该分片中的批次数量
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # 去掉最后一个不完整的批次
                assert num_batches > 0, "这个分片文件太小了？请检查。"
                # 随机打乱批次索引
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                # 对每个批次生成输入 x 和目标输出 y
                for ix in ixs:
                    start = ix * self.max_seq_len  # 批次起始索引
                    end = start + self.max_seq_len + 1  # 批次结束索引
                    # 将数据转换为 NumPy 数组并拷贝到 RAM 中
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    # 模型输入 x 是当前批次的前 max_seq_len 个词元
                    x = chunk[:-1]
                    # 模型输出 y 是下一个词元
                    y = chunk[1:]
                    # 生成 x, y 对
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    pretokenize(vocab_size=4096)