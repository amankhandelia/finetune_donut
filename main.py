from functools import partial
import json
import random
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

import optax
from flax.training.train_state import TrainState
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from jax_smi import initialise_tracking
from tqdm.auto import tqdm

from datasets import load_dataset
from PIL import Image
from ast import literal_eval

initialise_tracking()
print(jax.devices())

dataset = load_dataset("nielsr/rvl_cdip_10_examples_per_class_donut")

id2label = {id: label for id, label in enumerate(dataset['train'].features['label'].names)}
print(id2label)

example = dataset["train"][0]
example["ground_truth"]


literal_eval(example["ground_truth"])['gt_parse']

from transformers import VisionEncoderDecoderConfig

max_length = 8
# image_size = [2560, 1920]
# let's use a smaller image size (height, width) because otherwise OOM
# the higher the resolution, the better the results will be
# so if you have a big GPU, feel free to increase
image_size = [512, 512]
height, width = image_size
num_channels = 3

# update image_size of the encoder
# during pre-training, a larger image size was used
config = VisionEncoderDecoderConfig.from_pretrained("nielsr/donut-base")
config.encoder.image_size = image_size  # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length
# TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602

from transformers import (BartConfig, DonutProcessor,
                          FlaxVisionEncoderDecoderModel)

processor = DonutProcessor.from_pretrained("nielsr/donut-base")
model = FlaxVisionEncoderDecoderModel.from_pretrained("nielsr/donut-base", config=config, from_pt=True)

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
processor.feature_extractor.size = image_size[::-1]  # should be (width, height)
processor.feature_extractor.do_align_long_axis = False

"""## Prepare dataset

The first thing we'll do is add the class names as added tokens to the vocabulary of the decoder of Donut, and the corresponding tokenizer.

This will result in a slight increase in performance, as otherwise a class might be split up into multiple subword tokens (e.g. the class "advertisement" might be split up into "adv", "ertisement"). It is beneficial to let the model just learn a single embedding vector for the token "advertisement".
"""

len(processor.tokenizer)

"""Hack #1

In order to add the add new token to the embedding layer I am directly replacing the old embedding layer with a newer one and than calling model.init with the newer model params
"""

additional_tokens = [
    "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", "<form/>", "<handwritten/>", "<invoice/>",
    "<letter/>", "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", "<resume/>",
    "<scientific_publication/>", "<scientific_report/>", "<specification/>"]

seed = jax.random.PRNGKey(seed=42)
extra_tokens = jax.random.normal(seed, shape=(len(additional_tokens), 1024))

newly_added_num = processor.tokenizer.add_tokens(additional_tokens)

model.params['decoder']['model']['decoder']['embed_tokens']['embedding'] = jnp.concatenate(
    [model.params['decoder']['model']['decoder']['embed_tokens']['embedding'], extra_tokens], axis=0)

assert model.params['decoder']['model']['decoder']['embed_tokens']['embedding'].shape[0] == len(processor.tokenizer)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_rvlcdip>'])[0]

len(processor.tokenizer)


class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in additional_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:

            seed = jax.random.PRNGKey(seed=42)
            extra_tokens = jax.random.normal(seed, shape=(len(list_of_tokens), 1024))

            model.params['decoder']['model']['decoder']['embed_tokens']['embedding'] = jnp.concatenate(
                [model.params['decoder']['model']['decoder']['embed_tokens']['embedding'], extra_tokens], axis=0)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # pixel values (we remove the batch dimension)
        pixel_values = processor(
            sample["image"].convert("RGB"),
            random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # labels, which are the input ids of the target sequence
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)

        encoding = dict(pixel_values=pixel_values,
                        labels=labels)

        return encoding


def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = np.asarray(input_ids[:, :-1])
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)

    return np.asarray(shifted_input_ids, dtype=np.int32)


def collate_donut_dataset(item_ls: List[Dict[str, np.ndarray]], device_count: int,
                          pad_token_id: int, decoder_start_token_id: int, for_pmap: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                                                                          np.ndarray, np.ndarray]:
    images = []
    labels = []
    for item in item_ls:
        images.append(item['pixel_values'].numpy())
        labels.append(item['labels'].numpy())
    images = np.stack(images)
    labels = np.stack(labels)
    # images, labels = jnp.asarray(batch["pixel_values"].numpy()), batch["labels"].numpy()
    images = images.transpose(0, 2, 3, 1)
    decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)
    labels = np.asarray(labels)
    position_ids = np.broadcast_to(np.arange(0, labels.shape[1]), labels.shape)

    if for_pmap:
        images = np.reshape(images, (device_count, -1, *images.shape[1:]))
        decoder_input_ids = np.reshape(decoder_input_ids, (device_count, -1, *decoder_input_ids.shape[1:]))
        labels = np.reshape(labels, (device_count, -1, *labels.shape[1:]))
        position_ids = np.reshape(position_ids, (device_count, -1, *position_ids.shape[1:]))

    return images, labels, decoder_input_ids, position_ids


train_dataset = DonutDataset("nielsr/rvl_cdip_10_examples_per_class_donut", max_length=max_length,
                             split="train", task_start_token="<s_rvlcdip>", prompt_end_token="<s_rvlcdip>",
                             sort_json_key=False,  # rvlcdip dataset is preprocessed, so no need for this
                             )

# I'm using a small batch size to make sure it fits in the memory Colab provides
train_batch_size = jax.local_device_count() * 4
collate_fn = partial(collate_donut_dataset, device_count=jax.local_device_count(),
                     pad_token_id=model.config.pad_token_id, decoder_start_token_id=model.config.decoder_start_token_id)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, drop_last=True, collate_fn=collate_fn, prefetch_factor=2, num_workers=2)


validation_dataset = DonutDataset("nielsr/rvl_cdip_10_examples_per_class_donut", max_length=max_length,
                                  split="test", task_start_token="<s_rvlcdip>", prompt_end_token="<s_rvlcdip>",
                                  sort_json_key=False,  # rvlcdip dataset is preprocessed, so no need for this
                                  )
# I'm using a small batch size to make sure it fits in the memory Colab provides
validation_batch_size = 8
val_collate_fn = partial(collate_donut_dataset, device_count=jax.local_device_count(),
                         pad_token_id=model.config.pad_token_id,
                         decoder_start_token_id=model.config.decoder_start_token_id, for_pmap=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size,
                                   shuffle=True, drop_last=False, collate_fn=val_collate_fn, prefetch_factor=2,
                                   num_workers=2)


def sanity_check():
    # batch = next(iter(train_dataloader))

    # for id in batch['labels'][0].tolist():
    #     if id != -100:
    #         print(processor.decode([id]))
    #     else:
    #         print(id)

    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)

    # # unnormalize
    # reconstructed_image = (batch['pixel_values'][0] * torch.tensor(std)[:, None, None]) + torch.tensor(mean)[:, None, None]
    # # unrescale
    # reconstructed_image = reconstructed_image * 255
    # # convert to numpy of shape HWC
    # reconstructed_image = torch.moveaxis(reconstructed_image, 0, -1)
    # image = Image.fromarray(reconstructed_image.numpy().astype(np.uint8))
    pass


# sanity check
print("Pad token ID:", processor.decode([model.config.pad_token_id]))
print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

seed = jax.random.PRNGKey(seed=1729)

input_shape = (
    (1, height, width, num_channels),
    (1, 1),
)
model.config.decoder.vocab_size = 57545
model.params = model.init_weights(seed, input_shape, model.params)


def train_step(state: TrainState, batch):
    images, labels, decoder_input_ids, position_ids = batch
    images, labels, decoder_input_ids, position_ids = jnp.array(images), jnp.array(
        labels), jnp.array(decoder_input_ids), jnp.array(position_ids)

    def loss_fn(params: dict, state: TrainState, images: jnp.ndarray, decoder_input_ids: jnp.ndarray,
                position_ids: jnp.ndarray, labels: jnp.ndarray):
        lm_outputs = state.apply_fn({'params': params}, images, decoder_input_ids, None, position_ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=lm_outputs.logits[:, :4], labels=labels[:, :4]).mean()
        return loss, lm_outputs.logits

    # TODO: investigate why is allow_int necessary
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)

    # Use pmap to parallelize the computation
    pmap_fn = jax.pmap(gradient_fn, in_axes=(None, None, 0, 0, 0, 0), axis_name='batch')
    (loss, logits), grads = pmap_fn(state.params, state, images, decoder_input_ids, position_ids, labels)
    grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    state = state.apply_gradients(grads=grads)

    return loss, state


lr = 1e-5
state = TrainState.create(
    apply_fn=model.module.apply,
    params=model.params,
    tx=optax.adamw(lr)
)


def get_args_for_models(batch: dict, model: FlaxVisionEncoderDecoderModel):
    images, labels = jnp.asarray(batch["pixel_values"].numpy()), batch["labels"].numpy()
    images = images.transpose(0, 2, 3, 1)
    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    labels = jnp.asarray(labels)
    position_ids = jnp.broadcast_to(jnp.arange(0, labels.shape[1]), labels.shape)
    return images, labels, decoder_input_ids, position_ids


def get_accuracy_score(
        model: FlaxVisionEncoderDecoderModel, state: TrainState, val_dataloader: DataLoader, verbose: bool = False) -> int:
    results = []

    for i, batch in enumerate(tqdm(val_dataloader)):

        # get args for model
        images, labels, decoder_input_ids, position_ids = batch
        images, labels, decoder_input_ids, position_ids = jnp.array(images), jnp.array(
            labels), jnp.array(decoder_input_ids), jnp.array(position_ids)

        # classify doc
        lm_outputs = model.module.apply({'params': state.params}, images, decoder_input_ids, None, position_ids)
        y_hat = jnp.argmax(lm_outputs.logits, axis=-1)

        if verbose:
            print(labels, y_hat)

        # ignore pad tokens and evaluate all others
        result = jnp.all(jnp.where(labels != -100, y_hat == labels, True), axis=-1)
        results.append(jnp.ravel(result))

    return jnp.stack(results).sum() / len(val_dataloader.dataset)


def get_accuracy_score_generate(
        model: FlaxVisionEncoderDecoderModel, state: TrainState, val_dataset: DonutDataset, verbose: bool = False) -> int:
    import re
    task_prompt = "<s_rvlcdip>"
    output_list = []
    accs = []
    for idx, sample in tqdm(enumerate(val_dataset.dataset), total=len(val_dataset)):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="np").pixel_values
        pixel_values = pixel_values
        # prepare decoder inputs

        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="np").input_ids

        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            max_length=8,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            decoder_start_token_id=decoder_input_ids,
        )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor.token2json(seq)

        ground_truth = json.loads(sample["ground_truth"])
        gt = ground_truth["gt_parse"]
        score = float(seq["class"] == gt["class"]) if 'class' in seq else 0.

        accs.append(score)

        output_list.append(seq)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")

    return


verbose = False
for epoch in range(30):
    print("Epoch:", epoch + 1)
    losses = []
    for i, batch in enumerate(tqdm(train_dataloader)):
        loss, state = train_step(state, batch)
        losses.append(jnp.mean(loss))
    print(f"Loss: {jnp.mean(jnp.stack(losses))}")
    if epoch >= 15:
        verbose = False
    model.params = state.params
    # score = get_accuracy_score(model, state, validation_dataloader, verbose=verbose)
    score = get_accuracy_score_generate(model, state, validation_dataset, verbose=verbose)
    print(f"Accuracy: {score}")
