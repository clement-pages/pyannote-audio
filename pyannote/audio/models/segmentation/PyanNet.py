# MIT License
#
# Copyright (c) 2020 CNRS
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

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.sincnet_old import SincNetOld
from pyannote.audio.utils.params import merge_dict


class PyanNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1, "per_channel": True}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        sincnet["num_channels"] = num_channels
        if not "channel_groups" in sincnet:
            sincnet["channel_groups"] = {str(i): [i] for i in range(num_channels)}
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)

        monolithic = lstm["monolithic"]
        self.lstm = nn.ModuleList()
        for i in range(len(sincnet["channel_groups"])):
            if monolithic:
                multi_layer_lstm = dict(lstm)
                del multi_layer_lstm["monolithic"]
                # self.lstm = nn.LSTM(60*num_channels, **multi_layer_lstm)
                self.lstm.append(nn.LSTM(60, **multi_layer_lstm))

            else:
                num_layers = lstm["num_layers"]
                if num_layers > 1:
                    self.dropout = nn.Dropout(p=lstm["dropout"])

                one_layer_lstm = dict(lstm)
                one_layer_lstm["num_layers"] = 1
                one_layer_lstm["dropout"] = 0.0
                del one_layer_lstm["monolithic"]

                self.lstm.append(nn.ModuleList(
                    [
                        nn.LSTM(
                            60
                            if i == 0
                            else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                            **one_layer_lstm
                        )
                        for i in range(num_layers)
                    ]
                ))

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features * len(sincnet["channel_groups"]),
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)
        if len(outputs.shape) == 3:
            outputs = torch.unsqueeze(outputs, 2)
        
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm):
            group_outputs = outputs[:, :, i, :]
            group_outputs = rearrange(group_outputs, "batch feature frame -> batch frame feature")
            if self.hparams.lstm["monolithic"]:
                group_outputs, _ = lstm(group_outputs)
            else:
                for i, lstm_layer in enumerate(lstm):
                    group_outputs, _ = lstm_layer(group_outputs)
                    if i + 1 < self.hparams.lstm["num_layers"]:
                        group_outputs = self.dropout(group_outputs)
            lstm_outputs.append(group_outputs)
        
        outputs = torch.cat(lstm_outputs, dim=-1)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))


class PyanNetMultiSincNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1, "per_channel": True}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnetMC = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnetMC["sample_rate"] = sample_rate
        sincnetMC["num_channels"] = num_channels - 1
        sincnetMC["channel_groups"] = {"accelerometre": [0, 1, 2]}
        sincnetSC = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnetSC["sample_rate"] = sample_rate
        sincnetSC["num_channels"] = num_channels
        sincnetSC["channel_groups"] = {str(i):[i] for i in range(num_channels)}
        # if not "channel_groups" in sincnet:
        #     sincnet["channel_groups"] = {str(i): [i] for i in range(num_channels)}
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("lstm", "linear")

        self.sincnetMC = SincNet(**sincnetMC)
        self.sincnetSC = SincNet(**sincnetSC)

        self.sincnet = self.sincnetMC

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            # self.lstm = nn.LSTM(60*num_channels, **multi_layer_lstm)
            self.lstm = nn.LSTM((len(sincnetSC["channel_groups"]) + 1) * 60, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputsMC = self.sincnetMC(waveforms[:, :3, :])
        outputsMC = torch.unsqueeze(outputsMC, 2)

        outputsSC = self.sincnetSC(waveforms)

        outputs = torch.cat((outputsSC, outputsMC), dim=2)
        outputs = rearrange(outputs, "batch feature channel frame -> batch frame (feature channel)")
        outputs = F.layer_norm()
        if self.hparams.lstm["monolithic"]:
            #print("Tenseur a l'entrée = ",t.shape)
            outputs, _ = self.lstm(outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))


class PyanNetOld(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        self.sincnet0 = SincNetOld(**self.hparams.sincnet)
        self.sincnet1 = SincNetOld(**self.hparams.sincnet)
        self.sincnet2 = SincNetOld(**self.hparams.sincnet)
        self.sincnet3 = SincNetOld(**self.hparams.sincnet)

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(60*num_channels, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.sincnet0.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.sincnet0.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.sincnet0.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        # print("taille waveforms (pour info : (batch, channel, sample)) = ",waveforms.shape)
        outputs0 = self.sincnet0(waveforms[:,0:1,:])
        outputs1 = self.sincnet1(waveforms[:,1:2,:])
        outputs2 = self.sincnet2(waveforms[:,2:3,:])
        outputs3 = self.sincnet3(waveforms[:,3:4,:])
        outputs=torch.cat((outputs0, outputs1, outputs2, outputs3), 1)
        outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
        
        if self.hparams.lstm["monolithic"]:
            #print("Tenseur a l'entrée = ",t.shape)
            outputs, _ = self.lstm(outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))


class PyanNetLateFusion(Model):
    """PyanNet segmentation model with late fusion (one LSTM for each sincnet outputs, e.g. one LSTM for each sensor)

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1, "per_channel": True}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        sincnet["num_channels"] = num_channels
        if not "channel_groups" in sincnet:
            sincnet["channel_groups"] = {str(i): [i] for i in range(num_channels)}
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)

        monolithic = lstm["monolithic"]
        self.lstm = nn.ModuleList()
        for i in range(len(sincnet["channel_groups"])):
            if monolithic:
                multi_layer_lstm = dict(lstm)
                del multi_layer_lstm["monolithic"]
                # self.lstm = nn.LSTM(60*num_channels, **multi_layer_lstm)
                self.lstm.append(nn.LSTM(60, **multi_layer_lstm))

            else:
                num_layers = lstm["num_layers"]
                if num_layers > 1:
                    self.dropout = nn.Dropout(p=lstm["dropout"])

                one_layer_lstm = dict(lstm)
                one_layer_lstm["num_layers"] = 1
                one_layer_lstm["dropout"] = 0.0
                del one_layer_lstm["monolithic"]

                self.lstm.append(nn.ModuleList(
                    [
                        nn.LSTM(
                            60
                            if i == 0
                            else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                            **one_layer_lstm
                        )
                        for i in range(num_layers)
                    ]
                ))

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features * len(sincnet["channel_groups"]),
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)
        if len(outputs.shape) == 3:
            outputs = torch.unsqueeze(outputs, 2)
        
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm):
            group_outputs = outputs[:, :, i, :]
            group_outputs = rearrange(group_outputs, "batch feature frame -> batch frame feature")
            if self.hparams.lstm["monolithic"]:
                group_outputs, _ = lstm(group_outputs)
            else:
                for i, lstm_layer in enumerate(lstm):
                    group_outputs, _ = lstm_layer(group_outputs)
                    if i + 1 < self.hparams.lstm["num_layers"]:
                        group_outputs = self.dropout(group_outputs)
            lstm_outputs.append(group_outputs)
        
        outputs = torch.cat(lstm_outputs, dim=-1)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))