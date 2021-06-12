# Revised by Haoyu Song 2020
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Encoder-Decoder architectures """

import torch

from typing import Optional
from torch import argmax

from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .modeling_outputs import Seq2SeqLMOutput
from .modeling_utils import PreTrainedModel
from .utils import logging



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-tsequencece model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    :meth:`~transformers.AutoModel.from_pretrained` function and the decoder is loaded via
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` function. Cross-attention layers are automatically added
    to the decoder and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks
    <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for sequence to sequence training to the decoder. Indices can be obtained using
            :class:`~transformers.PretrainedTokenizer`. See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (:obj:`tuple(torch.FloatTensor)`, `optional`):
            This tuple must consist of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
            sequence_length, hidden_size)`) is a tensor of hidden-states at the output of the last layer of the
            encoder. Used in the cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert :obj:`decoder_input_ids`
            indices into associated vectors than the model's internal embedding lookup matrix.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in ``[-100, 0,
            ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.Seq2SeqLMOutput` instead of a
            plain tuple.
        kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as ``**encoder_kwargs`` for the encoder forward function.
            - With a `decoder_` prefix which will be input as ``**decoder_kwargs`` for the decoder forward function.
"""


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class EncoderDecoderModel(PreTrainedModel):

    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        decoder2: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder is not None and decoder2 is not None
        ), "Either a configuration or an Encoder and two decoders has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            from .modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from .modeling_auto import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        if decoder2 is None:
            from .modeling_auto import AutoModelForCausalLM

            decoder2 = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder
        self.decoder2 = decoder2
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_decoder2(self):
        return self.decoder2

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        decoder2_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        kwargs_decoder2 = {
            argument[len("decoder2_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder2_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]
        for key in kwargs_decoder2.keys():
            del kwargs["decoder2_" + key]

        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModel

            if "config" not in kwargs_encoder:
                from .configuration_auto import AutoConfig

                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder:
                from .configuration_auto import AutoConfig

                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        decoder2 = kwargs_decoder2.pop("model", None)
        if decoder2 is None:
            assert (
                decoder2_pretrained_model_name_or_path is not None
            ), "If `decoder2_model` is not defined as an argument, a `decoder2_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder2:
                from .configuration_auto import AutoConfig

                decoder2_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder2_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder2_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder2_config.is_decoder = True
                    decoder2_config.is_decoder2 = True
                    decoder2_config.add_cross_attention = True

                kwargs_decoder2["config"] = decoder2_config

            if kwargs_decoder2["config"].is_decoder2 is False or kwargs_decoder2["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder2 model {decoder_pretrained_model_name_or_path} is not initialized as a decoder2. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )
            decoder2 = AutoModelForCausalLM.from_pretrained(decoder2_pretrained_model_name_or_path, **kwargs_decoder2)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, decoder2=decoder2, config=config)

    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_generated_outputs=None,
        decoder2_inputs_embeds=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,  # TODO: (PVP) implement :obj:`use_cache`
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        eval_ppl=False,
        training=False,
        stage2=False,
        ul_training=False,
        inference_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("encoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        kwargs_decoder2 = {
            argument[len("decoder2_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder2_")
        }

        split_index = int(token_type_ids.sum(-1)[0].detach().data)

        persona_input_ids = input_ids[:, :split_index]
        query_input_ids = input_ids[:, split_index:]

        persona_attention_mask = attention_mask[:, :split_index]
        query_attention_mask = attention_mask[:, split_index:]


        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                per_input_ids=persona_input_ids,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]
        encoder_embeddings = encoder_outputs[2][0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            per_input_ids=persona_input_ids,
            **kwargs_decoder,
        )

        decoder_hidden_states = decoder_outputs.hidden_states[-1]
        decoder2_inputs_embeds = decoder_outputs.hidden_states[0]

        decoder2_input_ids = argmax(decoder_outputs.logits, dim=-1)

        decoder2_attention_mask_extended = torch.ones(decoder2_input_ids.shape).cuda().masked_fill_(decoder2_input_ids==0,0)

        if training or eval_ppl:
            decoder2_outputs = self.decoder2(
                input_ids= decoder_input_ids,
                attention_mask= decoder_attention_mask,
                encoder_hidden_states= decoder_hidden_states,
                encoder_attention_mask=decoder2_attention_mask_extended,
                inputs_embeds=None,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                per_input_ids=persona_input_ids,
                **kwargs_decoder2,
            )
            if ul_training:
                decoder_input_ids=inference_dict['neg_hyp_input_ids']
                hyp_attention_mask=inference_dict['neg_hyp_attention_mask']
                mask_flag = torch.Tensor.bool(1 - hyp_attention_mask)
                labels = decoder_input_ids.masked_fill(mask_flag, -100)
                persona_input_ids=inference_dict['neg_pre_input_ids']

                ul_outputs = self.decoder2(
                    input_ids=decoder_input_ids,
                    attention_mask=hyp_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    inputs_embeds=None,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    per_input_ids=persona_input_ids,
                    ul_training=ul_training,
                    **kwargs_decoder2,
                )

                decoder_input_ids = inference_dict['pos_hyp_input_ids']
                hyp_attention_mask = inference_dict['pos_hyp_attention_mask']
                mask_flag = torch.Tensor.bool(1 - hyp_attention_mask)
                labels = decoder_input_ids.masked_fill(mask_flag, -100)
                persona_input_ids = inference_dict['pos_pre_input_ids']

                ul_outputs_2 = self.decoder2(
                    input_ids=decoder_input_ids,
                    attention_mask=hyp_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    inputs_embeds=None,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    per_input_ids=persona_input_ids,
                    **kwargs_decoder2,
                )
                ul_outputs.loss = 0.4 * ul_outputs.loss + 0.6 * ul_outputs_2.loss
            else:
                ul_outputs = decoder2_outputs

        else:
            if stage2:
                assert decoder_generated_outputs is not None

                decoder2_attention_mask_extended = torch.ones([
                    decoder2_input_ids.shape[0],
                    decoder2_input_ids.shape[1],
                    decoder_generated_outputs.shape[1]]).to(torch.device('cuda'))

                decoder2_outputs = self.decoder2(
                    input_ids= decoder_input_ids,
                    attention_mask= None,
                    encoder_hidden_states= decoder_generated_outputs,
                    encoder_attention_mask=decoder2_attention_mask_extended,
                    inputs_embeds=None,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    per_input_ids=persona_input_ids,
                    **kwargs_decoder2,
                )
                ul_outputs = decoder2_outputs
            else:
                decoder2_outputs = decoder_outputs
                ul_outputs = decoder_outputs

        if not return_dict:
            return ul_outputs + decoder2_outputs + encoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss= decoder_outputs.loss if decoder_outputs.loss is not None and decoder2_outputs.loss is not None else None,
            logits=decoder_outputs.logits,
            past_key_values=None,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), Seq2SeqLMOutput(
            loss= decoder2_outputs.loss if decoder_outputs.loss is not None and decoder2_outputs.loss is not None else None,
            logits=decoder2_outputs.logits,
            past_key_values=None,
            decoder_hidden_states=decoder2_outputs.hidden_states,
            decoder_attentions=decoder2_outputs.attentions,
            cross_attentions=decoder2_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), Seq2SeqLMOutput(
            loss= ul_outputs.loss if ul_outputs.loss is not None and decoder2_outputs.loss is not None else None,
            logits=ul_outputs.logits,
            past_key_values=None,
            decoder_hidden_states=decoder2_outputs.hidden_states,
            decoder_attentions=decoder2_outputs.attentions,
            cross_attentions=decoder2_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, token_type_ids, past=None, attention_mask=None,
                                      persona_encoder_outputs=None, query_encoder_outputs=None,
                                      decoder2_generated_input_ids=None, encoder_input_ids=None, per_input_ids=None,
                                      **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "persona_encoder_outputs": persona_encoder_outputs,
            "query_encoder_outputs": query_encoder_outputs,
            "input_ids" : encoder_input_ids,
            "token_type_ids": token_type_ids,
            "per_input_ids": per_input_ids,
        }

        if "use_cache" in decoder_inputs:
            input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

        if "past_key_values" in decoder_inputs:
            input_dict["past_key_values"] = decoder_inputs["past_key_values"]

        if decoder2_generated_input_ids is not None:
            input_dict["decoder2_generated_input_ids"] = decoder2_generated_input_ids

        return input_dict

    def _reorder_cache(self, past, beam_idx):
        return self.decoder._reorder_cache(past, beam_idx)
