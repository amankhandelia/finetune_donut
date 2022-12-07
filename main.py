import jax
from jax_smi import initialise_tracking
initialise_tracking()

# something wrong is happening as for a single core all the memory is getting hogged
# try alternative ways of using pmap
# ref: https://colab.research.google.com/drive/1hXns2b6T8T393zSrKCSoUktye1YlSe8U
# device: Total 20.5GB
#           15.2GB (74.28%): TPU_0(process=0,(0,0,0,0))
#          771.9MB ( 3.67%): TPU_1(process=0,(0,0,0,1))
#          771.9MB ( 3.67%): TPU_2(process=0,(1,0,0,0))
#          771.9MB ( 3.67%): TPU_3(process=0,(1,0,0,1))
#          771.9MB ( 3.67%): TPU_4(process=0,(0,1,0,0))
#          771.9MB ( 3.67%): TPU_5(process=0,(0,1,0,1))
#          771.9MB ( 3.67%): TPU_6(process=0,(1,1,0,0))
#          771.9MB ( 3.67%): TPU_7(process=0,(1,1,0,1))

#  kind: Total 20.5GB
#           20.5GB (  100%): buffer
#        -625.0B (2.8e-06%): executable

import jax.numpy as jnp

print(jax.devices())

from datasets import load_dataset
from torch.utils.data import DataLoader
import json
import random
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset

dataset = load_dataset("nielsr/rvl_cdip_10_examples_per_class_donut")

id2label = {id: label for id, label in enumerate(dataset['train'].features['label'].names)}
print(id2label)

example = dataset["train"][0]
example["ground_truth"]

from ast import literal_eval

literal_eval(example["ground_truth"])['gt_parse']

from transformers import VisionEncoderDecoderConfig

max_length = 8
# image_size = [2560, 1920]
# let's use a smaller image size (height, width) because otherwise OOM
# the higher the resolution, the better the results will be
# so if you have a big GPU, feel free to increase
image_size = [400, 400]
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

from transformers import DonutProcessor, FlaxVisionEncoderDecoderModel, BartConfig

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


train_dataset = DonutDataset("nielsr/rvl_cdip_10_examples_per_class_donut", max_length=max_length,
                             split="train", task_start_token="<s_rvlcdip>", prompt_end_token="<s_rvlcdip>",
                             sort_json_key=False,  # rvlcdip dataset is preprocessed, so no need for this
                             )

# I'm using a small batch size to make sure it fits in the memory Colab provides
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)


validation_dataset = DonutDataset("nielsr/rvl_cdip_10_examples_per_class_donut", max_length=max_length,
                                  split="test", task_start_token="<s_rvlcdip>", prompt_end_token="<s_rvlcdip>",
                                  sort_json_key=False,  # rvlcdip dataset is preprocessed, so no need for this
                                  )

# I'm using a small batch size to make sure it fits in the memory Colab provides
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


batch = next(iter(train_dataloader))
print(batch.keys())

for id in batch['labels'][0].tolist():
    if id != -100:
        print(processor.decode([id]))
    else:
        print(id)

from PIL import Image
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# unnormalize
reconstructed_image = (batch['pixel_values'][0] * torch.tensor(std)[:, None, None]) + torch.tensor(mean)[:, None, None]
# unrescale
reconstructed_image = reconstructed_image * 255
# convert to numpy of shape HWC
reconstructed_image = torch.moveaxis(reconstructed_image, 0, -1)
image = Image.fromarray(reconstructed_image.numpy().astype(np.uint8))
image

"""## Train model

Ok there's one additional thing before we can start training the model: during training, the model can create the `decoder_input_ids` (the decoder inputs) automatically based on the `labels` (by simply shifting them one position to the right, prepending the `decoder_start_token_id` and replacing labels which are -100 by the `pad_token_id`). Therefore, we need to set those variables, to make sure the `decoder_input_ids` are created automatically.

This ensures we only need to prepare labels for the model. Theoretically you can also create the `decoder_input_ids` yourself and not set the 2 variables below. This is what the original authors of Donut did.
"""

import optax
from flax.training.train_state import TrainState

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_rvlcdip>'])[0]

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

    return jnp.asarray(shifted_input_ids, dtype=jnp.int32)


def train_step(state: TrainState, batch):
    device_count = jax.local_device_count()

    images, labels = jnp.asarray(batch["pixel_values"].numpy()), batch["labels"].numpy()
    images = images.transpose(0, 2, 3, 1)
    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    labels = jnp.asarray(labels)
    position_ids = jnp.broadcast_to(jnp.arange(0, labels.shape[1]), labels.shape)
    images = jnp.reshape(images, (device_count, -1, *images.shape[1:]))
    decoder_input_ids = jnp.reshape(decoder_input_ids, (device_count, -1, *decoder_input_ids.shape[1:]))
    labels = jnp.reshape(labels, (device_count, -1, *labels.shape[1:]))
    position_ids = jnp.reshape(position_ids, (device_count, -1, *position_ids.shape[1:]))

    def loss_fn(params, images, decoder_input_ids, position_ids, labels):
        lm_outputs = state.apply_fn({'params': params}, images, decoder_input_ids, None, position_ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=lm_outputs.logits, labels=labels).mean()
        return loss, lm_outputs.logits

    # hack #2 I am using allow_int to handle this model param, which is not be finetuned
    # https://github.com/amankhandelia/transformers/blob/feature/donut_flax_implementation/src/transformers/models/donut/modeling_flax_donut_swin.py#L411
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)

    # Use pmap to parallelize the computation
    pmap_fn = jax.pmap(gradient_fn, in_axes=(None, 0, 0, 0, 0))
    (loss, logits), grads = pmap_fn(state.params, images, decoder_input_ids, position_ids, labels)
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


def get_accuracy_score(model: FlaxVisionEncoderDecoderModel, state: TrainState, val_dataloader: DataLoader) -> int:
    results = []

    for i, batch in enumerate(tqdm(val_dataloader)):
        # get args for model
        images, labels, decoder_input_ids, position_ids = get_args_for_models(batch, model)

        # classify doc
        lm_outputs = model.module.apply({'params': state.params}, images, decoder_input_ids, None, position_ids)
        y_hat = jnp.argmax(lm_outputs.logits, axis=-1)

        # ignore pad tokens and evaluate all others
        result = jnp.all(jnp.select(labels != -100, y_hat == labels, True), axis=-1)
        results.append(jnp.ravel(result))

    return jnp.stack(results).sum() / len(val_dataloader.dataset)


from tqdm.auto import tqdm
for epoch in range(10):
    print("Epoch:", epoch + 1)
    for i, batch in enumerate(tqdm(train_dataloader)):
        loss, state = train_step(state, batch)
    print(f"Loss: {loss}")

    score = get_accuracy_score(model, state, validation_dataloader)
    print(f"Accuracy: {score}")
