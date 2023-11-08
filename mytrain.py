import argparse
import os
import subprocess
import sys
from itertools import count
from multiprocessing import Process

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import *

from dataset import Corpus, EncodedDataset, ValidEncodedDataset
from download import download, ALL_DATASETS
from utils import summary, distributed
from transformers.adapters import MAMConfig

'''
lastinput = []
def hook(module, fea_in, fea_out):
    lastinput.append(fea_in[0].data)
    return None
'''


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        output_d[key] = tensor_input.item()
    return output_d


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, epoch_size=None, token_dropout=None, seed=None):
    # download(real_dataset, fake_dataset, data_dir=data_dir)

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)
    real_train, real_valid = real_corpus.train, real_corpus.valid
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    min_sequence_length = None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                   token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    validation_dataset = ValidEncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_loader, validation_loader


def train(model, optimizer, device: str, loader: DataLoader, desc='Train'):
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    with tqdm(loader, desc=desc) as loop:
        for tokens1, mask1, label1, tokens2, mask2, label2 in loop:
            texts = torch.cat((tokens1, tokens2), dim=0)
            masks = torch.cat((mask1, mask2), dim=0)
            labels = torch.cat((label1, label2), dim=0)
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]
            optimizer.zero_grad()
            out = model(texts, attention_mask=masks, labels=labels, output_hidden_states=True, return_dict=True)

            dis_loss = 0
            '''
            print(lastinput)
            true_feat = lastinput[0][:batch_size//2]
            false_feat = lastinput[0][batch_size//2:]
            for i in range(len(true_feat)):
                dis = torch.cosine_similarity(true_feat[i],false_feat)
                dis_loss += 0.2 - dis[i] + min(dis)
            dis_loss /= batch_size

            lastinput.pop()

            print(len(out.hidden_states))
            print(out.hidden_states[-1].shape)
            print(lastinput[0].shape, out.hidden_states[-1][:, 0, :].shape)
            print(lastinput[0],out.hidden_states[-1][:,0,:])
            '''
            feat = out.hidden_states[-1][:, 0, :]
            true_feat = feat[:batch_size // 2]
            false_feat = feat[batch_size // 2:]
            for i in range(len(false_feat)):
                dis = torch.cosine_similarity(true_feat[i], false_feat)
                dis_loss += 0.2 - dis[i] + min(dis)
            dis_loss /= batch_size

            loss = out.loss + 0.1 * dis_loss
            logits = out.logits
            # out.loss.backward()
            loss.backward()
            optimizer.step()

            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, device: str, loader: DataLoader, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    with tqdm(loader, desc=desc) as loop, torch.no_grad():
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            out = model(texts, attention_mask=masks, labels=labels)
            loss = out.loss
            logits = out.logits

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)
    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }


def main(batch_size=32,
         max_sequence_length=128,
         large=False,
         device="cuda:0",
         data_dir='data',
         real_dataset='webtext',
         fake_dataset='xl-1542M-k40',
         epoch_size=None,
         token_dropout=None,
         seed=None,
         epoch_loop=range(10),
         learning_rate=2e-5,
         weight_decay=1e-8,
         **kwargs):
    torch.manual_seed(seed)
    model_name = 'roberta-large' if large else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    config = MAMConfig()
    model.add_adapter("mam_adapter", config=config)
    model.train_adapter("mam_adapter")
    model.set_active_adapters("mam_adapter")
    model.to(device)
    # list(model.modules())[-3].register_forward_hook(hook=hook)

    train_loader, validation_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, epoch_size,
                                                    token_dropout, seed)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter("ZCH_Tensorboard_Trying_logs")
    best_validation_accuracy = 0
    for epoch in epoch_loop:
        train_metrics = train(model, optimizer, device, train_loader, f'Epoch {epoch}')
        validation_metrics = validate(model, device, validation_loader)

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        for key, value in combined_metrics.items():
            writer.add_scalar(key, value, global_step=epoch)

        if combined_metrics["validation/accuracy"] > best_validation_accuracy:
            best_validation_accuracy = combined_metrics["validation/accuracy"]

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(dict(
                epoch=epoch,
                model_state_dict=model_to_save.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            ),
                "best-model.pt"
            )


if __name__ == "__main__":
    download(*ALL_DATASETS)
    main(seed=0)
