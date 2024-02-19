import itertools
import warnings
from collections import defaultdict
from pathlib import Path
from tempfile import mkstemp
from typing import Dict, Literal, Optional, Text, Tuple, Union

import numpy as np
import torch
from pyannote.core import Segment
from pyannote.database import Protocol
from torch.utils.data._utils.collate import default_collate
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAUROC

from pyannote.audio.core.task import (
    Problem,
    Resolution,
    Scopes,
    Specifications,
    Subsets,
    get_dtype,
)
from pyannote.audio.tasks.segmentation.mixins import Task
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.random import create_rng_for_worker


# Your custom task must be a subclass of `pyannote.audio.core.task.Task`
class MultiClassSegmentation(Task):
    """Sound event detection"""

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 1.0,
        min_duration: float = 5.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation=None,
        weight: Optional[Text] = None,
        remove_pad: bool = False,
        cache: Optional[Union[str, None]] = None,
        **other_params,
    ):

        super().__init__(
            protocol,
            duration=duration,
            min_duration=min_duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            cache=cache,
        )
        self.weight = weight
        self.remove_pad = remove_pad

    def prepare_data(self):
        """Use this to prepare data from task protocol

        Notes
        -----
        Called only once on the main process (and only on it), for global_rank 0.

        After this method is called, the task should have a `prepared_data` attribute
        with the following dictionary structure:

        prepared_data = {
            'protocol': name of the protocol
            'audio-path': array of N paths to audio
            'audio-metadata': array of N audio infos such as audio subset, scope and database
            'audio-info': array of N audio torchaudio.info struct
            'audio-encoding': array of N audio encodings
            'audio-annotated': array of N annotated duration (usually equals file duration but might be shorter if file is not fully annotated)
            'annotations-regions': array of M annotated regions
            'annotations-segments': array of M' annotated segments
            'metadata-values': dict of lists of values for subset, scope and database
            'metadata-`database-name`-labels': array of `database-name` labels. Each database with "database" scope labels has it own array.
            'metadata-labels': array of global scope labels
        }

        """

        if self.cache:
            # check if cache exists and is not empty:
            if self.cache.exists() and self.cache.stat().st_size > 0:
                # data was already created, nothing to do
                return
            # create parent directory if needed
            self.cache.parent.mkdir(parents=True, exist_ok=True)
        else:
            # if no cache was provided by user, create a temporary file
            # in system directory used for temp files
            self.cache = Path(mkstemp()[1])

        # list of possible values for each metadata key
        # (will become .prepared_data[""])
        metadata_unique_values = defaultdict(list)
        metadata_unique_values["subset"] = Subsets
        metadata_unique_values["scope"] = Scopes

        audios = list()  # list of path to audio files
        audio_infos = list()
        audio_encodings = list()
        metadata = list()  # list of metadata

        annotated_duration = list()  # total duration of annotated regions (per file)
        annotated_regions = list()  # annotated regions
        annotations = list()  # actual annotations
        unique_labels = list()
        database_unique_labels = {}

        if self.has_validation:
            files_iter = itertools.chain(
                self.protocol.train(), self.protocol.development()
            )
        else:
            files_iter = self.protocol.train()

        for file_id, file in enumerate(files_iter):
            # gather metadata and update metadata_unique_values so that each metadatum
            # (e.g. source database or label) is represented by an integer.
            metadatum = dict()

            # keep track of source database and subset (train, development, or test)
            if file["database"] not in metadata_unique_values["database"]:
                metadata_unique_values["database"].append(file["database"])
            metadatum["database"] = metadata_unique_values["database"].index(
                file["database"]
            )
            metadatum["subset"] = Subsets.index(file["subset"])

            # keep track of label scope (file, database, or global)
            metadatum["scope"] = Scopes.index(file["scope"])

            remaining_metadata_keys = set(file) - set(
                [
                    "uri",
                    "database",
                    "subset",
                    "audio",
                    "torchaudio.info",
                    "scope",
                    "classes",
                    "annotation",
                    "annotated",
                ]
            )

            # keep track of any other (integer or string) metadata provided by the protocol
            # (e.g. a "domain" key for domain-adversarial training)
            for key in remaining_metadata_keys:
                value = file[key]

                if isinstance(value, str):
                    if value not in metadata_unique_values[key]:
                        metadata_unique_values[key].append(value)
                    metadatum[key] = metadata_unique_values[key].index(value)

                elif isinstance(value, int):
                    metadatum[key] = value

                else:
                    warnings.warn(
                        f"Ignoring '{key}' metadata because of its type ({type(value)}). Only str and int are supported for now.",
                        category=UserWarning,
                    )

            metadata.append(metadatum)

            # reset list of file-scoped labels
            file_unique_labels = list()

            # path to audio file
            audios.append(str(file["audio"]))

            # audio info
            audio_info = file["torchaudio.info"]
            audio_infos.append(
                (
                    audio_info.sample_rate,  # sample rate
                    audio_info.num_frames,  # number of frames
                    audio_info.num_channels,  # number of channels
                    audio_info.bits_per_sample,  # bits per sample
                )
            )
            audio_encodings.append(audio_info.encoding)  # encoding

            # annotated regions and duration
            _annotated_duration = 0.0
            for segment in file["annotated"]:
                # skip annotated regions that are shorter than training chunk duration
                # if segment.duration < self.duration:
                #    continue

                # append annotated region
                annotated_region = (
                    file_id,
                    segment.duration,
                    segment.start,
                )
                annotated_regions.append(annotated_region)

                # increment annotated duration
                _annotated_duration += segment.duration

            # append annotated duration
            annotated_duration.append(_annotated_duration)

            # annotations
            for segment, _, label in file["annotation"].itertracks(yield_label=True):
                # "scope" is provided by speaker diarization protocols to indicate
                # whether speaker labels are local to the file ('file'), consistent across
                # all files in a database ('database'), or globally consistent ('global')

                # 0 = 'file' / 1 = 'database' / 2 = 'global'
                scope = Scopes.index(file["scope"])

                # update list of file-scope labels
                if label not in file_unique_labels:
                    file_unique_labels.append(label)
                # and convert label to its (file-scope) index
                file_label_idx = file_unique_labels.index(label)

                database_label_idx = global_label_idx = -1

                if scope > 0:  # 'database' or 'global'
                    # update list of database-scope labels
                    database = file["database"]
                    if database not in database_unique_labels:
                        database_unique_labels[database] = []
                    if label not in database_unique_labels[database]:
                        database_unique_labels[database].append(label)

                    # and convert label to its (database-scope) index
                    database_label_idx = database_unique_labels[database].index(label)

                if scope > 1:  # 'global'
                    # update list of global-scope labels
                    if label not in unique_labels:
                        unique_labels.append(label)
                    # and convert label to its (global-scope) index
                    global_label_idx = unique_labels.index(label)

                annotations.append(
                    (
                        file_id,  # index of file
                        segment.start,  # start time
                        segment.end,  # end time
                        file_label_idx,  # file-scope label index
                        database_label_idx,  # database-scope label index
                        global_label_idx,  # global-scope index
                    )
                )

        # since not all metadata keys are present in all files, fallback to -1 when a key is missing
        metadata = [
            tuple(metadatum.get(key, -1) for key in metadata_unique_values)
            for metadatum in metadata
        ]
        metadata_dtype = [
            (key, get_dtype(max(m[i] for m in metadata)))
            for i, key in enumerate(metadata_unique_values)
        ]

        # turn list of files metadata into a single numpy array
        # TODO: improve using https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
        info_dtype = [
            (
                "sample_rate",
                get_dtype(max(ai[0] for ai in audio_infos)),
            ),
            (
                "num_frames",
                get_dtype(max(ai[1] for ai in audio_infos)),
            ),
            ("num_channels", "B"),
            ("bits_per_sample", "B"),
        ]

        # turn list of annotated regions into a single numpy array
        region_dtype = [
            (
                "file_id",
                get_dtype(max(ar[0] for ar in annotated_regions)),
            ),
            ("duration", "f"),
            ("start", "f"),
        ]

        # turn list of annotations into a single numpy array
        segment_dtype = [
            (
                "file_id",
                get_dtype(max(a[0] for a in annotations)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("file_label_idx", get_dtype(max(a[3] for a in annotations))),
            ("database_label_idx", get_dtype(max(a[4] for a in annotations))),
            ("global_label_idx", get_dtype(max(a[5] for a in annotations))),
        ]

        # save all protocol data in a dict
        prepared_data = {}

        # keep track of protocol name
        prepared_data["protocol"] = self.protocol.name

        prepared_data["audio-path"] = np.array(audios, dtype=np.str_)
        audios.clear()

        prepared_data["audio-metadata"] = np.array(metadata, dtype=metadata_dtype)
        metadata.clear()

        prepared_data["audio-info"] = np.array(audio_infos, dtype=info_dtype)
        audio_infos.clear()

        prepared_data["audio-encoding"] = np.array(audio_encodings, dtype=np.str_)
        audio_encodings.clear()

        prepared_data["audio-annotated"] = np.array(annotated_duration)
        annotated_duration.clear()

        prepared_data["annotations-regions"] = np.array(
            annotated_regions, dtype=region_dtype
        )
        annotated_regions.clear()

        prepared_data["annotations-segments"] = np.array(
            annotations, dtype=segment_dtype
        )
        annotations.clear()

        prepared_data["metadata-values"] = metadata_unique_values

        for database, labels in database_unique_labels.items():
            prepared_data[f"metadata-{database}-labels"] = np.array(
                labels, dtype=np.str_
            )
        database_unique_labels.clear()

        prepared_data["metadata-labels"] = np.array(unique_labels, dtype=np.str_)
        unique_labels.clear()

        self.prepare_validation(prepared_data)
        self.post_prepare_data(prepared_data)

        # save prepared data on the disk
        with open(self.cache, "wb") as cache_file:
            np.savez_compressed(cache_file, **prepared_data)

    def post_prepare_data(self, prepared_data: Dict):
        """Method for completing `prepared_data` with task-specific data.
        For instance, for a classification task, this could be a list of
        possible classes.

        Parameters
        ----------
        prepared_data: dict
            dictionnary containing protocol data prepared by
            `prepare_data()`
        Note
        ----
        This method does not return anything. Thus, user have to directly modify
        `prepared_data`, for updates to be taken into account
        """
        pass

    def setup(self, stage=None):
        """Setup data cached by prepare_data into the task on each device"""

        # send cache path on all processes used for the training,
        # allowing them to access the cache generated by prepare_data
        if stage == "fit":
            self.cache = self.trainer.strategy.broadcast(self.cache)

        try:
            with open(self.cache, "rb") as cache_file:
                self.prepared_data = dict(np.load(cache_file, allow_pickle=True))
        except FileNotFoundError:
            print(
                "Cached data for protocol not found. Ensure that prepare_data() was called",
                " and executed correctly or/and that the path to the task cache is correct.",
            )
            raise

        # checks that the task current protocol matches the cached protocol
        if self.protocol.name != self.prepared_data["protocol"]:
            raise ValueError(
                f"Protocol specified for the task ({self.protocol.name}) "
                f"does not correspond to the cached one ({self.prepared_data['protocol']})"
            )

    def post_prepare_data(self, prepared_data: Dict):
        # this method is called at the end of Task.prepare_data()
        # to complete data preparation with task-specific data, here
        # the list of classes and some training metadata

        # load metadata for training subset
        prepared_data["train_metadata"] = list()
        for training_file in self.protocol.train():
            prepared_data["train_metadata"].append(
                {
                    # path to audio file (str)
                    "audio": training_file["audio"],
                    # duration of audio file (float)
                    "duration": training_file["torchaudio.info"].num_frames
                    / training_file["torchaudio.info"].sample_rate,
                    # reference annotation (pyannote.core.Annotation)
                    "annotation": training_file["annotation"],
                }
            )

        # gather the list of classes
        classes = set()
        for training_file in prepared_data["train_metadata"]:
            classes.update(training_file["annotation"].labels())
        prepared_data["classes"] = sorted(classes)

        # `has_validation` is True if protocol defines a development set
        if not self.has_validation:
            return

    def prepare_validation(self, prepared_data: Dict):

        # load metadata for validation subset
        prepared_data["validation"] = list()
        for validation_file in self.protocol.development():
            prepared_data["validation"].append(
                {
                    "audio": validation_file["audio"],
                    "num_samples": validation_file["torchaudio.info"].num_frames,
                    "annotation": validation_file["annotation"],
                }
            )

    def setup(self, stage: Optional[Union[str, None]] = None):
        # this method assigns prepared data from task.prepare_data() to the task
        # and declares the task specifications

        super().setup(stage)
        # specify the addressed problem
        self.specifications = Specifications(
            # it is a multi-label classification problem
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            # we expect the model to output one prediction
            # for the whole chunk
            resolution=Resolution.CHUNK,
            # the model will ingest chunks with that duration (in seconds)
            duration=self.duration,
            # human-readable names of classes
            classes=self.prepared_data["classes"],
        )

    def default_metric(self):
        num_classes = len(self.specifications.classes)
        return {
            "MulticlassAUROC": MulticlassAUROC(
                num_classes=num_classes, average="macro", compute_on_cpu=True
            ),
            "F1": F1Score(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=-1,
                average="macro",
            ),
            "Precision": Precision(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=-1,
                average="macro",
            ),
            "Recall": Recall(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=-1,
                average="macro",
            ),
            "Accuracy": Accuracy(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=-1,
                average="macro",
            ),
        }

    def train__iter__(self):
        rng = create_rng_for_worker(self.model)

        # load list and number of classes
        classes = list(self.specifications.classes)
        num_classes = len(classes)

        # yield training samples "ad infinitum"
        while True:

            # select training file at random
            random_training_file, *_ = rng.choices(
                self.prepared_data["train_metadata"], k=1
            )

            # select one chunk at random
            random_start_time = rng.uniform(
                0, random_training_file["duration"] - self.duration
            )
            random_chunk = Segment(random_start_time, random_start_time + self.duration)

            # load audio excerpt corresponding to random chunk
            X, pad_start, pad_end = self.model.audio.crop(
                random_training_file["audio"],
                random_chunk,
                duration=self.duration,
                mode="pad",
            )

            # load labels corresponding to random chunk as {0|1} numpy array
            # y[k] = 1 means that kth class is active
            y = np.zeros((num_classes,))
            active_classes = (
                random_training_file["annotation"].crop(random_chunk).labels()
            )
            for active_class in active_classes:
                y[classes.index(active_class)] = 1
            # yield training samples as a dict (use 'X' for input and 'y' for target)
            yield {"X": X[0], "y": y, "pad": torch.tensor([pad_start, pad_end])}

    def collate_fn(self, batch, stage="train"):
        collated = default_collate(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated["X"],
            sample_rate=self.model.hparams.sample_rate,
        )
        collated["X"] = augmented.samples

        return collated

    def common_step(self, batch, batch_idx, stage: Literal["train", "val"]):
        # forward pass
        if self.remove_pad:
            y_pred = self.model(batch["X"], num_pad=batch["pad"])
        else:
            y_pred = self.model(batch["X"])

        batch_size, _, num_classes = y_pred.shape
        # (batch_size, 1, num_classes)

        # target
        y = batch["y"]
        # (batch_size, num_classes)
        y = torch.where(y == 1)[1]
        # (batch_size,)

        # class weight determined from classes distribution in training partition of the challenge dataset
        # A C D F H N S U
        if stage == "train":
            class_weight = 1000 / torch.tensor(
                [3053, 2243, 1426, 1139, 13440, 25106, 3882, 2897],
                device=self.model.device,
            )
        else:
            class_weight = None
        # compute loss
        loss = nll_loss(y_pred, y, class_weight=class_weight)

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            f"loss/{stage}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if stage == "val":
            self.model.validation_metric(y_pred.squeeze(1), y)

            self.model.log_dict(
                self.model.validation_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "val")

    def train__len__(self):
        # since train__iter__ runs "ad infinitum", we need a way to define what an epoch is.
        # this is the purpose of this method. it outputs the number of training samples that
        # make an epoch.

        # we compute this number as the total duration of the training set divided by
        # duration of training chunks. we make sure that an epoch is at least one batch long,
        # or pytorch-lightning will complain

        return max(self.batch_size, len(self.prepared_data["train_metadata"]))

    def val__getitem__(self, sample_idx):
        # load list and number of classes
        classes = list(self.specifications.classes)
        num_classes = len(classes)

        validation_file = self.prepared_data["validation"][sample_idx]

        chunk = Segment(start=0.0, end=self.duration)
        # load audio excerpt corresponding to current chunk
        X, pad_start, pad_end = self.model.audio.crop(
            validation_file["audio"], chunk, duration=self.duration, mode="pad"
        )

        # load labels corresponding to random chunk as {0|1} numpy array
        # y[k] = 1 means that kth class is active
        y = np.zeros((num_classes,))
        active_classes = validation_file["annotation"].crop(chunk).labels()
        for active_class in active_classes:
            y[classes.index(active_class)] = 1

        return {"X": X[0], "y": y, "pad": torch.tensor([pad_start, pad_end])}

    def val__len__(self):
        return len(self.prepared_data["validation"])
