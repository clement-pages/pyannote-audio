# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict
import copy
import itertools
import math
from pathlib import Path
import random
import warnings
from tempfile import mkstemp
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from einops import rearrange
from matplotlib import pyplot as plt
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_metric_learning.losses import ArcFaceLoss
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import (
    Problem, Resolution, Specifications, get_dtype
)
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker

Subtask = Literal["diarization", "embedding"]

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)
Subtasks = list(Subtask.__args__)


class JointSpeakerDiarizationAndEmbedding(SpeakerDiarization):
    """Joint speaker diarization and embedding task

    Usage
    -----
    load a meta protocol containing both diarization (e.g. X.SpeakerDiarization.Pretraining)
    and verification (e.g. VoxCeleb.SpeakerVerification.VoxCeleb) datasets
    >>> from pyannote.database import registry
    >>> protocol = registry.get_protocol(...)

    instantiate task
    >>> task = JointSpeakerDiarizationAndEmbedding(protocol)

    instantiate multi-task model
    >>> model = JointSpeakerDiarizationAndEmbeddingModel()
    >>> model.task = task

    train as usual...

    """

    def __init__(
        self,
        protocol,
        duration: float = 5.0,
        max_speakers_per_chunk: int = 3,
        max_speakers_per_frame: int = 2,
        weigh_by_cardinality: bool = False,
        batch_size: int = 32,
        dia_task_rate: float = 0.5,
        num_workers: int = None,
        pin_memory: bool = False,
        margin: float = 28.6,
        scale: float = 64.0,
        alpha: float = 0.5,
        augmentation: BaseWaveformTransform = None,
        cache: Optional[Union[str, None]] = None,
    ) -> None:
        """TODO Add docstring"""
        super().__init__(
            protocol,
            duration=duration,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            weigh_by_cardinality=weigh_by_cardinality,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            cache=cache,
        )

        self.num_dia_samples = int(batch_size * dia_task_rate)
        self.margin = margin
        self.scale = scale
        self.alpha = alpha
        # keep track of the use of database available in the meta protocol
        # * embedding databases are those with global speaker label scope
        # * diarization databases are those with file or database speaker label scope
        self.global_files_id = []
        self.database_dataset_idxs = []

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
                zip(itertools.repeat("train"), self.protocol.train()),
                zip(itertools.repeat("development"), self.protocol.development()),
            )
        else:
            files_iter = zip(itertools.repeat("train"), self.protocol.train())

        for file_id, (subset, file) in enumerate(files_iter):
            # gather metadata and update metadata_unique_values so that each metadatum
            # (e.g. source database or label) is represented by an integer.
            metadatum = dict()

            # keep track of source database and subset (train, development, or test)
            if file["database"] not in metadata_unique_values["database"]:
                metadata_unique_values["database"].append(file["database"])
            metadatum["database"] = metadata_unique_values["database"].index(
                file["database"]
            )

            metadatum["subset"] = Subsets.index(subset)

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
                    # continue

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

        if self.has_validation:
            self.prepare_validation(prepared_data)

        self.post_prepare_data(prepared_data)

        # save prepared data on the disk
        with open(self.cache, "wb") as cache_file:
            np.savez_compressed(cache_file, **prepared_data)

    def setup(self, stage="fit"):
        """Setup method

        Parameters
        ----------
        stage : {'fit', 'validate', 'test'}, optional
            Setup stage. Defaults to 'fit'.
        """

        super().setup()

        self.prepared_data["metadata-values"] = (
            self.prepared_data["metadata-values"].tolist()
        )

        global_scope_mask = (
            self.prepared_data["annotations-segments"]["global_label_idx"] > -1
        )
        self.global_files_id = np.unique(
            self.prepared_data["annotations-segments"]["file_id"][global_scope_mask]
        )

        mask = (
            self.prepared_data["audio-metadata"][:]["scope"] > 0
        )
        database_indexes = (
            np.unique(self.prepared_data["audio-metadata"][mask]["database"])
        )
        classes = {}
        for database_index in database_indexes:
            database = self.prepared_data["metadata-values"]["database"][database_index]
            classes[database] = [i for i in range(len(self.prepared_data[f"metadata-{database}-labels"]))]

        # if there is no file dedicated to the embedding task
        if self.alpha != 1.0 and len(classes) == 0:
            self.num_dia_samples = self.batch_size
            self.alpha = 1.0
            warnings.warn(
                "No class found for the speaker embedding task. Model will be trained on the speaker diarization task only."
            )

        if self.alpha != 0.0 and np.sum(global_scope_mask) == len(
            self.prepared_data["annotations-segments"]
        ):
            self.num_dia_samples = 0
            self.alpha = 0.0
            warnings.warn(
                "No segment found for the speaker diarization task. Model will be trained on the speaker embedding task only."
            )

        speaker_diarization = Specifications(
            duration=self.duration,
            resolution=Resolution.FRAME,
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            permutation_invariant=True,
            classes=[f"speaker{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
        )
        speaker_embedding = Specifications(
            duration=self.duration,
            resolution=Resolution.CHUNK,
            problem=Problem.REPRESENTATION,
            classes=classes,
        )
        self.specifications = (speaker_diarization, speaker_embedding)

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target as a SlidingWindowFeature instance where y.labels is
                   in meta.scope space.
            - `meta`:
                - `scope`: target scope (0: file, 1: database, 2: global)
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        # get label scope
        label_scope = Scopes[self.prepared_data["audio-metadata"][file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(
            file, chunk, duration=duration, mode="pad"
        )

        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start - half
        end_idx = np.round(end / step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start:end + 1, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(y, self.model.receptive_field, labels=labels)

        metadata = self.prepared_data["audio-metadata"][file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def draw_diarization_chunk(
        self,
        file_ids: np.ndarray,
        cum_prob_annotated_duration: np.ndarray,
        rng: random.Random,
        duration: float,
    ) -> tuple:
        """Sample one chunk for the diarization task

        Parameters
        ----------
        file_ids: np.ndarray
            array containing files id
        cum_prob_annotated_duration: np.ndarray
            array of the same size than file_ids array, containing probability
            to corresponding file to be drawn
        rng : random.Random
            Random number generator
        duration: float
            duration of the chunk to draw
        """
        # select one file at random (wiht probability proportional to its annotated duration)
        file_id = file_ids[cum_prob_annotated_duration.searchsorted(rng.random())]
        # find indices of annotated regions in this file
        annotated_region_indices = np.where(
            self.prepared_data["annotations-regions"]["file_id"] == file_id
        )[0]

        # turn annotated regions duration into a probability distribution
        cum_prob_annotaded_regions_duration = np.cumsum(
            self.prepared_data["annotations-regions"]["duration"][
                annotated_region_indices
            ]
            / np.sum(
                self.prepared_data["annotations-regions"]["duration"][
                    annotated_region_indices
                ]
            )
        )

        # seletect one annotated region at random (with probability proportional to its duration)
        annotated_region_index = annotated_region_indices[
            cum_prob_annotaded_regions_duration.searchsorted(rng.random())
        ]

        # select one chunk at random in this annotated region
        _, region_duration, start = self.prepared_data["annotations-regions"][
            annotated_region_index
        ]
        start_time = rng.uniform(start, start + region_duration - duration)

        return (file_id, start_time)

    def draw_embedding_chunk(self, database: str, class_idx: int, duration: float) -> tuple:
        """Sample one chunk for the embedding task

        Parameters
        ----------
        database: string
            database from which draw embedding chunk
        class_idx : int
            class index in the task speficiations
        duration: float
            duration of the chunk to draw

        Return
        ------
        tuple:
            file_id:
                the file id to which the sampled chunk belongs
            start_time:
                start time of the sampled chunk
        """

        # get all files for current database
        database_idx = self.prepared_data["metadata-values"]["database"].index(database)
        database_files_id = np.argwhere(
            self.prepared_data["audio-metadata"][:]["database"] == database_idx
        ).flatten()
        database_mask = np.isin(self.prepared_data["annotations-segments"]["file_id"], database_files_id)

        # get all segments for the current classes
        class_mask = self.prepared_data["annotations-segments"]["database_label_idx"] == class_idx

        # get segments for current class = segments which belong to current database and class
        class_segments = self.prepared_data["annotations-segments"][np.logical_and(database_mask, class_mask)]

        # sample one segment from all the class segments:
        segments_duration = class_segments["end"] - class_segments["start"]
        segments_total_duration = np.sum(segments_duration)
        prob_segments = segments_duration / segments_total_duration
        segment = np.random.choice(class_segments, p=prob_segments)

        # sample chunk start time in order to intersect it with the sampled segment
        start_time = np.random.uniform(
            max(segment["start"] - duration, 0), segment["end"]
        )

        return (segment["file_id"], start_time)

    def train__iter__helper(self, rng: random.Random, **filters):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key : value} dict), filter training files so that
            only file such as file [key] == value are used for generating chunks

        Yields
        ------
        chunk : dict
        Training chunks
        """

        # indices of training files that matches domain filters
        training = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "train"
        )
        for key, value in filters.items():
            training &= self.prepared_data["audio-metadata"][key] == self.prepared_data[
                "metadata"
            ][key].index(value)
        file_ids = np.where(training)[0]
        # get the subset of embedding database files from training files
        embedding_files_ids = file_ids[np.isin(file_ids, self.global_files_id)]

        if self.num_dia_samples > 0:
            annotated_duration = self.prepared_data["audio-annotated"][file_ids]
            # set duration of files for the embedding part to zero, in order to not
            # drawn them for diarization part
            annotated_duration[embedding_files_ids] = 0.0

            cum_prob_annotated_duration = np.cumsum(
                annotated_duration / np.sum(annotated_duration)
            )

        duration = self.duration
        batch_size = self.batch_size

        # use original order for the first run on the shuffled classes list:
        classes = copy.deepcopy(
            self.specifications[Subtasks.index("embedding")].classes
        )

        sample_idx = 0
        database = rng.choice(self.prepared_data["metadata-values"]["database"])
        database_classes_idx = {database: 0 for database in classes}
        while True:
            if sample_idx < self.num_dia_samples:
                file_id, start_time = self.draw_diarization_chunk(
                    file_ids, cum_prob_annotated_duration, rng, duration
                )
            else:
                # shuffle database classes list and go through this shuffled list
                # to make sure to see all the database's speakers during training
                if database_classes_idx[database] == len(classes[database]):
                    rng.shuffle(classes[database])
                    database_classes_idx[database] = 0

                klass = classes[database][database_classes_idx[database]]
                database_classes_idx[database] += 1

                file_id, start_time = self.draw_embedding_chunk(database, klass, duration)

            sample = self.prepare_chunk(file_id, start_time, duration)

            sample_idx = (sample_idx + 1) % batch_size
            if sample_idx == 0:
                rng.choice(self.prepared_data["metadata-values"]["database"])

            yield sample

    def train__iter__(self):
        """Iterate over trainig samples

        Yields
        ------
        dict:
            x: (time, channel)
                Audio chunks.
            task: "diarization" or "embedding"
            y: target speaker label for speaker embedding task,
                (frame, ) frame-level targets for speaker diarization task.
                Note that frame < time.
                `frame is infered automagically from the exemple model output`
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.model)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)
        else:
            # create
            subchunks = dict()
            for product in itertools.product(
                [self.prepared_data["metadata-values"][key] for key in balance]
            ):
                filters = {key: value for key, value in zip(balance, product)}
                subchunks[product] = self.train__iter__helper(rng, **filters)

        while True:
            # select one subchunck generator at random (with uniform probability)
            # so thath it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(subchunks)]

            # generate random chunk
            yield next(chunks)

    def collate_y(self, batch) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : list
            List of samples to collate.
            "y" field is expected to be a SlidingWindowFeature.

        Returns
        -------
        y : torch.Tensor
            Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
            If one chunk has more than `self.max_speakers_per_chunk` speakers, we keep
            the max_speakers_per_chunk most talkative ones. If it has less, we pad with
            zeros (artificial inactive speakers).
        """

        collated_y_dia = []
        collate_y_emb = []

        for b in batch:
            # diarization reference
            y_dia = b["y"].data
            labels = b["y"].labels
            num_speakers = len(labels)
            # embedding reference
            y_emb = np.full((self.max_speakers_per_chunk,), -1, dtype=int)

            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y_dia, axis=0), axis=0)
                # keep only the most talkative speakers
                y_dia = y_dia[:, indices[: self.max_speakers_per_chunk]]
                # TODO: we should also sort the speaker labels in the same way

                # if current chunck is for the embedding subtask
                if b["meta"]["scope"] > 0:
                    labels = np.array(labels)
                    y_emb = labels[indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                # create inactive speakers by zero padding
                y_dia = np.pad(
                    y_dia,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )
                if b["meta"]["scope"] > 0:
                    y_emb[:num_speakers] = labels[:]

            else:
                if b["meta"]["scope"] > 0:
                    y_emb[:num_speakers] = labels[:]

            collated_y_dia.append(y_dia)
            collate_y_emb.append(y_emb)

        return (
            torch.from_numpy(np.stack(collated_y_dia)),
            torch.from_numpy(np.stack(collate_y_emb)).squeeze(1),
        )

    def collate_fn(self, batch, stage="train"):
        """Collate function used for most segmentation tasks

        This function does the following:
        * stack waveforms into a (batch_size, num_channels, num_samples) tensor batch["X"])
        * apply augmentation when in "train" stage
        * convert targets into a (batch_size, num_frames, num_classes) tensor batch["y"]
        * collate any other keys that might be present in the batch using pytorch default_collate function

        Parameters
        ----------
        batch : list of dict
            List of training samples.

        Returns
        -------
        batch : dict
            Collated batch as {"X": torch.Tensor, "y": torch.Tensor} dict.
        """

        # collate X
        collated_X = self.collate_X(batch)
        # collate y
        collated_y_dia, collate_y_emb = self.collate_y(batch)

        # collate metadata
        collated_meta = self.collate_meta(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y_dia.unsqueeze(1),
        )

        return {
            "X": augmented.samples,
            "y_dia": augmented.targets.squeeze(1),
            "y_emb": collate_y_emb,
            "meta": collated_meta,
        }

    def setup_loss_func(self):
        classes = self.specifications[Subtasks.index("embedding")].classes
        # define a loss for each database-scope dataset
        arcface_losses = []
        for database in classes:
            arcface_losses.append((
                database,
                ArcFaceLoss(
                        len(classes[database]),
                        self.model.hparams["embedding_dim"],
                        margin=self.margin,
                        scale=self.scale,
                    )
                )
            )
        self.model.arc_face_loss = nn.ModuleDict(
            {database: loss_func for database, loss_func in arcface_losses}
        )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        # `clamp_min` is needed to set non-speech weight to 1.
        class_weight = (
            torch.clamp_min(self.model.powerset.cardinality, 1.0)
            if self.weigh_by_cardinality
            else None
        )
        seg_loss = nll_loss(
            permutated_prediction,
            torch.argmax(target, dim=-1),
            class_weight=class_weight,
            weight=weight,
        )

        return seg_loss

    def compute_diarization_loss(self, prediction, permutated_target):
        """Compute loss for the speaker diarization subtask

        Parameters
        ----------
        prediction : torch.Tensor
            speaker diarization output predicted by the model for the current batch.
            Shape of (batch_size, num_spk, num_frames)
        permutated_target: torch.Tensor
            permutated target for the current batch. Shape of (batch_size, num_spk, num_frames)

        Returns
        -------
        dia_loss : torch.Tensor
            Permutation-invariant diarization loss
        """

        # Compute segmentation loss
        dia_loss = self.segmentation_loss(prediction, permutated_target)
        self.model.log(
            "loss/train/dia",
            dia_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return dia_loss

    def compute_embedding_loss(self, emb_prediction, target_emb, valid_embs, database: str):
        """Compute loss for the speaker embeddings extraction subtask

        Parameters
        ----------
        emb_prediction : torch.Tensor
            speaker embeddings predicted by the model for the current batch.
            Shape of (batch_size * num_spk, embedding_dim)
        target_emb : torch.Tensor
            target embeddings for the current batch
            Shape of (batch_size * num_spk,)
        Returns
        -------
        emb_loss : torch.Tensor
            arcface loss for the current batch
        """

        # Get speaker representations from the embedding subtask
        embeddings = rearrange(emb_prediction, "b s e -> (b s) e")
        # Get corresponding target label
        targets = rearrange(target_emb, "b s -> (b s)")
        # compute loss only on global scope speaker embedding
        valid_embs = rearrange(valid_embs, "b s -> (b s)")
        # compute the loss
        emb_loss = self.model.arc_face_loss[database](
            embeddings[valid_embs, :], targets[valid_embs]
        )

        if torch.any(valid_embs):
            emb_loss = (1. / torch.sum(valid_embs)) * emb_loss

        # skip batch if something went wrong for some reason
        if torch.isnan(emb_loss):
            return None

        self.model.log(
            f"loss/train/arcface_{database}",
            emb_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return emb_loss

    def training_step(self, batch, batch_idx: int):
        """Compute loss for the joint task

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        # batch waveforms (batch_size, num_channels, num_samples)
        waveform = batch["X"]
        # batch diarization references (batch_size, num_channels, num_speakers)
        target_dia = batch["y_dia"]
        # batch embedding references (batch, num_speakers)
        target_emb = batch["y_emb"]
        database = batch["meta"]["database"]

        # drop samples that contain too many speakers
        num_speakers = torch.sum(torch.any(target_dia, dim=1), dim=1)
        keep = num_speakers <= self.max_speakers_per_chunk

        target_dia = target_dia[keep]
        target_emb = target_emb[keep]
        waveform = waveform[keep]

        num_remaining_dia_samples = torch.sum(keep[: self.num_dia_samples])

        # corner case
        if not keep.any():
            return None

        # forward pass
        dia_prediction, emb_prediction = self.model(waveform)
        # (batch_size, num_frames, num_cls), (batch_size, num_spk, emb_size)

        # get the best permutation
        dia_multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target_dia, permut_map = permutate(dia_multilabel, target_dia)
        permutated_target_emb = target_emb[
            torch.arange(target_emb.shape[0]).unsqueeze(1), permut_map
        ]

        # an embedding is valid only if corresponding speaker is active in the diarization prediction and reference
        active_speaker_pred = torch.any(dia_multilabel > 0, dim=1)
        active_speaker_ref = torch.any(permutated_target_dia == 1, dim=1)
        valid_embs = torch.logical_and(active_speaker_pred, active_speaker_ref)[
            num_remaining_dia_samples:
        ]

        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target_dia.float()
        )

        dia_prediction = dia_prediction[:num_remaining_dia_samples]
        permutated_target_powerset = permutated_target_powerset[
            :num_remaining_dia_samples
        ]

        dia_loss = torch.tensor(0)
        # if batch contains diarization subtask chunks, then compute diarization loss on these chunks:
        if self.alpha != 0.0 and torch.any(keep[: self.num_dia_samples]):
            dia_loss = self.compute_diarization_loss(
                dia_prediction, permutated_target_powerset
            )

        emb_loss = torch.tensor(0)
        # if batch contains embedding subtask chunks, then compute embedding loss on these chunks:
        if self.alpha != 1.0 and torch.any(valid_embs):
            emb_prediction = emb_prediction[num_remaining_dia_samples:]
            permutated_target_emb = permutated_target_emb[num_remaining_dia_samples:]
            database_idx = batch["meta"]["database"][-1]
            database = self.prepared_data["metadata-values"]["database"][database_idx]
            emb_loss = self.compute_embedding_loss(
                emb_prediction, permutated_target_emb, valid_embs, database
            )
            loss = self.alpha * dia_loss + (1 - self.alpha) * emb_loss
        else:
            loss = self.alpha * dia_loss

        return {"loss": loss}

    # TODO: no need to compute gradient in this method
    def validation_step(self, batch, batch_idx: int):
        """Compute validation loss and metric

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        # target
        target_dia = batch["y_dia"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # TODO: should we handle validation samples with too many speakers
        # waveform = waveform[keep]
        # target = target[keep]

        # forward pass
        dia_prediction, _ = self.model(waveform)
        batch_size, num_frames, _ = dia_prediction.shape

        multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target, _ = permutate(multilabel, target_dia)

        # FIXME: handle case where target have too many speakers?
        # since we don't need
        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target.float()
        )
        seg_loss = self.segmentation_loss(
            dia_prediction,
            permutated_target_powerset,
        )

        self.model.log(
            "loss/val/dia",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target_dia, 1, 2),
        )

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # log first batch visualization every 2^n epochs.
        if (
            self.model.current_epoch == 0
            or math.log2(self.model.current_epoch) % 1 > 0
            or batch_idx > 0
        ):
            return

        # visualize first 9 validation samples of first batch in Tensorboard/MLflow

        y = permutated_target.float().cpu().numpy()
        y_pred = multilabel.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.NaN
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):
            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot target
            ax_ref = axes[row_idx * 2 + 0, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot predictions
            ax_hyp = axes[row_idx * 2 + 1, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

        plt.tight_layout()

        for logger in self.model.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("samples", fig, self.model.current_epoch)
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"samples_epoch{self.model.current_epoch}.png",
                )

        plt.close(fig)

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components for diarization subtask,
        and equal error rate for the embedding part
        """
        return {
            "DiarizationErrorRate": DiarizationErrorRate(0.5),
            "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
            "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
            "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
            # "EqualErrorRate": EqualErrorRate(compute_on_cpu=True, distances=False),
            # "BinaryAUROC": BinaryAUROC(compute_on_cpu=True),
        }
