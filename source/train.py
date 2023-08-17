# -*- coding:utf-8 -*- 
# coding:unicode_escape
# @Author: Lemon00
# @Time: 2023/8/11 11:13
# @File: train.py
import os
import time
from os.path import exists

import GPUtil
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import pad
from torch.utils.data import Dataset, DistributedSampler, DataLoader

from main import DummyOptimizer, DummyScheduler
from model import subsequent_mask, make_model
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtext.data import to_map_style_dataset


class Batch:
    def __init__(self, src, tgt=None, pad=1):  # -1 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 截取 0到 n-1的序列，丢弃最后一个输出
            self.tgt = tgt[:, :-1]
            # 截取 1 到 n 的序列
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    step: int = 0  # 当前轮数中的步数
    accum_step: int = 0  # 梯度累积步数
    samples: int = 0  # 使用的总样本数量
    tokens: int = 0  # 处理的总 token 数量


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """训练一个批次"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter #TODO: ? 为什么这么除
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            # 梯度累加器
            if i % accum_iter == 0:
                optimizer.step()
                # 清除累积梯度
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]
    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []
    for idx, example in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)
    learning_rates = torch.tensor(learning_rates)

    plt.figure(figsize=(10, 6))
    steps = np.arange(20000)
    for warmup_idx in [0, 1, 2]:
        plt.plot(steps, learning_rates[warmup_idx], label=f"model_size:warmup={opts[warmup_idx]}")

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.show()


# example_learning_schedule()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):  # TODO: 大概看明白了，就是没太明白
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = np.array(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    target = np.array([2, 1, 0, 3, 3])
    crit(torch.log(torch.tensor(predict)), torch.LongTensor(target))
    target_distribution = crit.true_dist.detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(target_distribution, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set tick labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(np.arange(5))
    ax.set_yticklabels(np.arange(5))
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    cbar.ax.set_ylabel('Target distribution')

    # Loop over data dimensions and create text annotations
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, np.round(target_distribution[i, j], decimals=2),
                           ha='center', va='center', color='w')

    plt.show()


# example_label_smoothing()


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


# Test
def data_gen(V, batch_size, nbatches):
    """
    :param V: 词汇表大小
    :param batch_size: 每个批次中样本的数量
    :param nbatches: 批次数量
    :return:
    """
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"data general in GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach().to(device)
        tgt = data.requires_grad_(False).clone().detach().to(device)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        # 预测结果
        x = self.generator(x)
        sloss = (
            # 计算损失值
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        # 计算的n个样本的损失值，所以要除以norm(也就是n)来平均损失
        return sloss.data * norm, sloss


# 贪婪解码，总是取概率最大值
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Train the simple copy task.
def example_simple_model_gpus():
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    # 词表大小
    V = 11
    batch_size = 80
    criterion = LabelSmoothing(size=V, padding_idx=1, smoothing=0.0).to(device)
    model = make_model(V, V, N=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# example_simple_model_gpus()
"""
vocab_size=65000,
bos_id=0,
pad_id=1,
eos_id=2,
unk_id=3,
"""


def load_tokenizers():
    try:
        # 英文分词器
        tokenizer_en = spm.SentencePieceProcessor(model_file='../data/pre/model-t-en.model')
    except IOError:
        print("Tokenizer_en can't load")
    try:
        # 中文分词器
        tokenizer_zh = spm.SentencePieceProcessor(model_file='../data/pre/model-t-zh.model')
    except IOError:
        print("Tokenizer_zh can't load")
    return tokenizer_zh, tokenizer_en


# 在这里的实现和原代码不一样，直接返回的就是类似于[29, 263, 103]的序列值，不需要再走一步词典的转换
# TODO:源代码
# def tokenize(text, tokenizer):
#     return [tok for tok in tokenizer.Encode(text)]

# TODO：测试代码
def tokenize(text, tokenizer):
    list = [tok for tok in tokenizer.Encode(text)]
    print(list)
    return list


# 本部分大幅改动
# 注意！去除了形参 src_vocab,tgt_vocab
def collate_batch(
        batch,
        tokenizer_zh,
        tokenizer_en,
        src_pipeline,
        tgt_pipeline,
        device,
        max_padding=128,
        pad_id=1,
):
    """
    :param tokenizer_zh: tokenize处理器
    :param tokenizer_en:
    :param batch: 批次数
    :param src_pipeline: tok
    :param tgt_pipeline: tgt_tokenize
    # :param src_vocab: 应该是将字符转换转换为数值的过程，由tokenize 函数实现，该参数已被删除
    # :param tgt_vocab: 应该是将字符转换转换为数值的过程，由tokenize 函数实现，该参数已被删除
    :param device: 设备数
    :param max_padding:
    :param pad_id:
    :return:返回了经过分词处理并数值化以后，并有相同长度的src,tgt列表
    """
    # <s> token id 这里tgt和src由于都使用的sentencePiece的默认配置所以特殊标记符号的位置相同
    bs_id = torch.tensor([tokenizer_zh.bos_id()], device=device)
    eos_id = torch.tensor([tokenizer_en.eos_id()], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                # 添加bs符号
                bs_id,
                torch.tensor(
                    # 将src先分词，再转换为数值
                    # src_vocab(src_pipeline(_src)) #原实现
                    src_pipeline(_src),  # 这里的实现在tokenize函数，直接返回了数值，不需要再走一步
                    dtype=torch.int64,
                    device=device
                ),
                eos_id  # 添加eos符号
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    # 同src处理方式
                    tgt_pipeline(_tgt),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - 如果输入长度大于最大填充长度，pad输入参数将变为负值
            pad(
                processed_src,
                (
                    0,
                    # 计算需要填充的数量
                    max_padding - len(processed_src)
                ),
                value=pad_id
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (
                    0,
                    max_padding - len(processed_tgt),
                ),
                value=pad_id
            )
        )
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


class mydataset(Dataset):
    def __init__(self, csv_file):  # self 参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        self.csv_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = self.csv_data.values[idx]
        return data


def create_dataloaders(
        device,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    """

    :param device:
    :param batch_size:
    :param max_padding:
    :param is_distributed: 分布式训练
    :return:
    """
    # 与原实现不同，但效果是一样的
    tokenizer_zh, tokenizer_en = load_tokenizers()

    def tokenize_en(text):
        return tokenize(text, tokenizer_en)

    def tokenize_zh(text):
        return tokenize(text, tokenizer_zh)

    def collate_fn(batch):
        #
        return collate_batch(
            batch,
            tokenizer_zh,
            tokenizer_en,
            tokenize_zh,
            tokenize_en,
            device,
            max_padding=max_padding,
            pad_id=1
        )

    train_iter = mydataset('../data/t/train.csv')
    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter = mydataset('../data/t/val.csv')
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        config,
        is_distributed=False
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
    # 这里选择先load tokenize 以继续模型的设置
    tokenizer_en, tokenizer_zh = load_tokenizers()
    pad_index = 1
    d_model = 512
    model = make_model(len(tokenizer_zh), len(tokenizer_en), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(tokenizer_zh), padding_idx=pad_index, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training =====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_index) for b in train_dataloader),  # TODO: dataloader数据出口
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch{epoch} Validation ===", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_index) for b in valid_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval"
        )
        print(loss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%s.pt" % config["file_prefix"]
        torch.save(model.state_dict(), file_path)


def train_distributed_model(config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    # 启动多进程达到分布式效果
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, config, True),
    )


def train_model(config):
    if config["distributed"]:
        train_distributed_model(
            config
        )
    else:
        train_worker(
            0, 1, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 512,
        "warmup": 3000,
        "file_prefix": "test_model"
    }
    model_path = "test_model.pt"
    if not exists(model_path):
        train_model(config)

    tokenizer_src, _ = load_tokenizers()
    model = make_model(len(tokenizer_src),len(tokenizer_src),N=6)
    model.load_state_dict(torch.load("test_model.pt"))
    return model


model = load_trained_model()