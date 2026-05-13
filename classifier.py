#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import os
import random, numpy as np, argparse
from collections import OrderedDict
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Model as HFGPT2Model
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False
OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.encoder_backend = getattr(config, 'encoder_backend', 'custom')
    if self.encoder_backend == 'custom':
      self.gpt = GPT2Model.from_pretrained()
      hidden_size = self.gpt.config.hidden_size
    elif self.encoder_backend == 'hf':
      self.gpt = HFGPT2Model.from_pretrained('gpt2')
      hidden_size = self.gpt.config.hidden_size
    else:
      raise ValueError(f"Unknown encoder backend: {self.encoder_backend}")

    # Pretrain mode does not require updating GPT paramters.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    pooled_size = hidden_size * 2
    head_hidden_size = getattr(config, 'head_hidden_size', hidden_size)
    self.classifier = torch.nn.Sequential(OrderedDict([
      ('layer_norm', torch.nn.LayerNorm(pooled_size)),
      ('dense', torch.nn.Linear(pooled_size, head_hidden_size)),
      ('activation', torch.nn.GELU()),
      ('dropout', torch.nn.Dropout(config.hidden_dropout_prob)),
      ('out_proj', torch.nn.Linear(head_hidden_size, self.num_labels))
    ]))



  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE

    # run gpt2 on the input sentences to get the contextualized embeddings
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

    if self.encoder_backend == 'custom':
      sequence_output = outputs["last_hidden_state"]
      last_token_hidden_state = outputs["last_token"]
    else:
      sequence_output = outputs.last_hidden_state
      last_non_pad_idx = attention_mask.sum(dim=1) - 1
      last_token_hidden_state = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]
    expanded_mask = attention_mask.unsqueeze(-1).float()
    summed_hidden_state = (sequence_output * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1.0)
    mean_hidden_state = summed_hidden_state / token_counts

    pooled_output = torch.cat([last_token_hidden_state, mean_hidden_state], dim=-1)
    logits = self.classifier(pooled_output)

    # verify that the output shape is correct (batch_size, num_labels)
    assert logits.shape == (input_ids.shape[0], self.num_labels)
    return logits
  



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.prompt_template = getattr(args, 'prompt_template', '{sentence}')

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [self.prompt_template.format(sentence=x[0]) for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.prompt_template = getattr(args, 'prompt_template', '{sentence}')

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [self.prompt_template.format(sentence=x[0]) for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def build_optimizer(model, args):
  no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight', 'ln_')
  optimizer_grouped_parameters = [
    {
      'params': [
        p for n, p in model.gpt.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
      ],
      'lr': args.lr,
      'weight_decay': args.weight_decay
    },
    {
      'params': [
        p for n, p in model.gpt.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
      ],
      'lr': args.lr,
      'weight_decay': 0.0
    },
    {
      'params': [
        p for n, p in model.classifier.named_parameters()
        if not any(nd in n for nd in no_decay)
      ],
      'lr': args.head_lr,
      'weight_decay': args.weight_decay
    },
    {
      'params': [
        p for n, p in model.classifier.named_parameters()
        if any(nd in n for nd in no_decay)
      ],
      'lr': args.head_lr,
      'weight_decay': 0.0
    }
  ]
  optimizer_grouped_parameters = [
    group for group in optimizer_grouped_parameters if len(group['params']) > 0
  ]
  return AdamW(optimizer_grouped_parameters, lr=args.lr)


def build_scheduler(optimizer, total_steps, warmup_ratio):
  warmup_steps = int(total_steps * warmup_ratio)

  def lr_lambda(current_step):
    if warmup_steps > 0 and current_step < warmup_steps:
      return float(current_step) / float(max(1, warmup_steps))
    return max(
      0.0,
      float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )

  return LambdaLR(optimizer, lr_lambda)


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'head_hidden_size': args.head_hidden_size,
            'data_dir': '.',
            'encoder_backend': args.encoder_backend,
            'fine_tune_mode': args.fine_tune_mode}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  optimizer = build_optimizer(model, args)
  total_steps = max(1, args.epochs * len(train_dataloader))
  scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, config, args.filepath)

    print(
      f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, "
      f"train f1 :: {train_f1 :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}"
    )


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--task", type=str, choices=('sst', 'cfimdb', 'both'), default='sst')
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="full-model")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for full-model fine-tuning: 1e-5",
                      default=1e-5)
  parser.add_argument("--head_lr", type=float, help="learning rate for the classifier head", default=5e-4)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--warmup_ratio", type=float, default=0.1)
  parser.add_argument("--head_hidden_size", type=int, default=768)
  parser.add_argument("--encoder_backend", type=str, choices=('custom', 'hf'), default='custom')
  parser.add_argument("--prompt_template", type=str, default='Review: {sentence}\nSentiment:')
  parser.add_argument("--no_prompt", action='store_true')

  args = parser.parse_args()
  return args


def make_task_config(args, task_name):
  is_sst = task_name == 'sst'
  batch_size = args.batch_size if is_sst else 8
  prompt_template = '{sentence}' if args.no_prompt else args.prompt_template
  prompt_name = 'noprompt' if args.no_prompt else 'prompt'
  return SimpleNamespace(
    filepath=os.path.join(
      CHECKPOINT_DIR,
      (
        f'{task_name}-{args.fine_tune_mode}-e{args.epochs}-lr{args.lr}'
        f'-headlr{args.head_lr}-{args.encoder_backend}-{prompt_name}-bs{batch_size}-seed{args.seed}.pt'
      )
    ),
    lr=args.lr,
    head_lr=args.head_lr,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=batch_size,
    hidden_dropout_prob=args.hidden_dropout_prob,
    head_hidden_size=args.head_hidden_size,
    encoder_backend=args.encoder_backend,
    prompt_template=prompt_template,
    train='data/ids-sst-train.csv' if is_sst else 'data/ids-cfimdb-train.csv',
    dev='data/ids-sst-dev.csv' if is_sst else 'data/ids-cfimdb-dev.csv',
    test='data/ids-sst-test-student.csv' if is_sst else 'data/ids-cfimdb-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out=os.path.join(
      PREDICTION_DIR,
      (
        f'{task_name}-{args.fine_tune_mode}-e{args.epochs}-lr{args.lr}'
        f'-headlr{args.head_lr}-{args.encoder_backend}-{prompt_name}-bs{batch_size}-seed{args.seed}-dev-out.csv'
      )
    ),
    test_out=os.path.join(
      PREDICTION_DIR,
      (
        f'{task_name}-{args.fine_tune_mode}-e{args.epochs}-lr{args.lr}'
        f'-headlr{args.head_lr}-{args.encoder_backend}-{prompt_name}-bs{batch_size}-seed{args.seed}-test-out.csv'
      )
    )
  )


def run_task(args, task_name):
  display_name = 'SST' if task_name == 'sst' else 'cfimdb'
  config = make_task_config(args, task_name)
  print(f'Training Sentiment Classifier on {display_name}...')
  train(config)
  print(f'Evaluating on {display_name}...')
  test(config)


if __name__ == "__main__":
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  os.makedirs(PREDICTION_DIR, exist_ok=True)
  args = get_args()
  seed_everything(args.seed)

  if args.task in ('sst', 'both'):
    run_task(args, 'sst')
  if args.task in ('cfimdb', 'both'):
    run_task(args, 'cfimdb')
