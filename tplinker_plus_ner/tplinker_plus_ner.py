import re
from tqdm import tqdm
import torch
import copy
import torch
import torch.nn as nn
import math
from common.components import HandshakingKernel


class HandshakingTaggingScheme(object):
    def __init__(self, entity_type2id, max_seq_len):
        super().__init__()

        self.separator = "\u2E80"

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags = {
            self.separator.join([ent, "EH2ET"]) for ent in self.ent2id.keys()
        }  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] 上三角矩阵
        self.shaking_idx2matrix_idx = [
            (ind, end_ind)
            for ind in range(self.matrix_size)
            for end_ind in list(range(self.matrix_size))[ind:]
        ]

        """
        最后的matrix_idx2shaking_idx是
        [[0, 1, 2]
         [0, 3, 4]
         [0, 0, 5]]
        """
        self.matrix_idx2shaking_idx = [
            [0 for i in range(self.matrix_size)] for j in range(self.matrix_size)
        ]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        """
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        """
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)

        #         # if entity_list exist, need to distinguish entity types
        #         if self.ent2id is not None and "entity_list" in sample:
        for ent in sample["entity_list"]:
            add_spot(
                (
                    ent["tok_span"][0],
                    ent["tok_span"][1] - 1,
                    self.tag2id[self.separator.join([ent["type"], "EH2ET"])],
                )
            )

        return matrix_spots

    def spots2shaking_tag(self, spots):
        """
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return: 
            shaking_tag: (shaking_seq_len, tag_size)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        """
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return: 
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(
            len(batch_spots), shaking_seq_len, len(self.tag2id)
        ).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        """
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_ent(self, text, shaking_tag, tok2char_span, tok_offset=0, char_offset=0):
        """
        shaking_tag: (shaking_seq_len, tag_id_num)
        """

        # matrix_spots:[(start_ind, end_ind, tag_id), ]
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            if (
                link_type != "EH2ET" or sp[0] > sp[1]
            ):  # for an entity, the start position can not be larger than the end pos.
                continue

            char_span_list = tok2char_span[sp[0] : sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0] : char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            head_key = str(sp[0])  # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # recover the positons in the original text
        for ent in ent_list:
            ent["char_span"] = [
                ent["char_span"][0] + char_offset,
                ent["char_span"][1] + char_offset,
            ]
            ent["tok_span"] = [
                ent["tok_span"][0] + tok_offset,
                ent["tok_span"][1] + tok_offset,
            ]

        return ent_list


class DataMaker4Bert:
    def __init__(self, tokenizer, shaking_tagger):
        self.tokenizer = tokenizer
        self.shaking_tagger = shaking_tagger

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(
            enumerate(data),
            total=len(data),
            desc="Generate indexed train or valid data",
        ):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                max_length=max_seq_len,
                truncation=True,
                pad_to_max_length=True,
            )

            # tagging
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(
                    sample
                )  # matrix_spots是将一个句子中所有的关系和实体整理成tagging的形式。[[token0,token1,tag],[token2,token3,tag]]，这里都是用数字表示了

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (
                sample,
                input_ids,
                attention_mask,
                token_type_ids,
                tok2char_span,
                matrix_spots,
            )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])
            if data_type != "test":
                matrix_spots_list.append(tp[5])

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(
                matrix_spots_list
            )  # batch_shaking_tag=((batch_size, shaking_seq_len, tag_size))

        return (
            sample_list,
            batch_input_ids,
            batch_attention_mask,
            batch_token_type_ids,
            tok2char_span_list,
            batch_shaking_tag,
        )


class DataMaker4BiLSTM:
    def __init__(self, text2indices, get_tok2char_span_map, shaking_tagger):
        self.text2indices = text2indices
        self.shaking_tagger = shaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(
            enumerate(data), desc="Generate indexed train or valid data"
        ):
            text = sample["text"]
            # tagging
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (
                sample,
                input_ids,
                tok2char_span,
                matrix_spots,
            )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            tok2char_span_list.append(tp[2])
            if data_type != "test":
                matrix_spots_list.append(tp[3])

        batch_input_ids = torch.stack(input_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(
                matrix_spots_list
            )

        return sample_list, batch_input_ids, tok2char_span_list, batch_shaking_tag


class TPLinkerPlusBert(nn.Module):
    def __init__(
        self, encoder, tag_size, shaking_type, inner_enc_type, tok_pair_sample_rate=1
    ):
        super().__init__()
        self.encoder = encoder
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = encoder.config.hidden_size

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(
            shaking_hidden_size, shaking_type, inner_enc_type
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        seq_len = last_hidden_state.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(
                shaking_hiddens.size()[0], 1
            )
            #             sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(
                shaking_hiddens.device
            )

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(
                1,
                sampled_tok_pair_indices[:, :, None].repeat(
                    1, 1, shaking_hiddens.size()[-1]
                ),
            )

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        outputs = self.fc(shaking_hiddens)

        return outputs, sampled_tok_pair_indices


class TPLinkerPlusBiLSTM(nn.Module):
    def __init__(
        self,
        init_word_embedding_matrix,
        emb_dropout_rate,
        enc_hidden_size,
        dec_hidden_size,
        rnn_dropout_rate,
        tag_size,
        shaking_type,
        inner_enc_type,
        tok_pair_sample_rate=1,
    ):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(
            init_word_embedding_matrix, freeze=False
        )
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(
            init_word_embedding_matrix.size()[-1],
            enc_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dec_lstm = nn.LSTM(
            enc_hidden_size,
            dec_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = dec_hidden_size

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(
            shaking_hidden_size, shaking_type, inner_enc_type
        )

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        seq_len = lstm_outputs.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, dec_hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(
                shaking_hiddens.size()[0], 1
            )
            #             sampled_tok_pair_indices = torch.randint(shaking_hiddens, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(
                shaking_hiddens.device
            )

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(
                1,
                sampled_tok_pair_indices[:, :, None].repeat(
                    1, 1, shaking_hiddens.size()[-1]
                ),
            )

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_hiddens, tag_size)
        outputs = self.fc(shaking_hiddens)
        return outputs, sampled_tok_pair_indices


class MetricsCalculator:
    def __init__(self, shaking_tagger):
        self.shaking_tagger = shaking_tagger
        self.last_weights = None  # for exponential moving averaging

    def GHM(self, gradient, bins=10, beta=0.9):
        """
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        """
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid(
            (gradient - avg) / std
        )  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(
            gradient_norm, 0, 0.9999999
        )  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        #         print()
        #         print("hits_vec: {}".format(hits_vec))
        #         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights
        #         print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(
            gradient_norm.size()[0], gradient_norm.size()[1], 1
        )
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true, ghm=True):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = (
            y_pred - (1 - y_true) * 1e12
        )  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def loss_func(self, y_pred, y_true, ghm):
        return self._multilabel_categorical_crossentropy(y_pred, y_true, ghm=ghm)

    def get_sample_accuracy(self, pred, truth):
        """
        计算该batch的pred与truth全等的样本比例
        """
        #         # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        #         pred = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, seq_len)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(
            correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]
        ).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc

    def get_mark_sets_event(self, event_list):
        trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set = (
            set(),
            set(),
            set(),
            set(),
        )
        for event in event_list:
            event_type = event["trigger_type"]
            trigger_offset = event["trigger_tok_span"]
            trigger_iden_set.add(
                "{}\u2E80{}".format(trigger_offset[0], trigger_offset[1])
            )
            trigger_class_set.add(
                "{}\u2E80{}\u2E80{}".format(
                    event_type, trigger_offset[0], trigger_offset[1]
                )
            )
            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                arg_iden_set.add(
                    "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                        event_type,
                        trigger_offset[0],
                        trigger_offset[1],
                        argument_offset[0],
                        argument_offset[1],
                    )
                )
                arg_class_set.add(
                    "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                        event_type,
                        trigger_offset[0],
                        trigger_offset[1],
                        argument_offset[0],
                        argument_offset[1],
                        argument_role,
                    )
                )

        return trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set

    def get_mark_sets_ent(self, ent_list, pattern="whole_text"):

        if pattern == "whole_span":
            ent_set = set(
                [
                    "{}\u2E80{}\u2E80{}".format(
                        ent["tok_span"][0], ent["tok_span"][1], ent["type"]
                    )
                    for ent in ent_list
                ]
            )

        elif pattern == "whole_text":
            ent_set = set(
                ["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in ent_list]
            )

        return ent_set

    def _cal_cpg(self, pred_set, gold_set, cpg):
        """
        cpg is a list: [correct_num, pred_num, gold_num]
        """
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

    def cal_ent_cpg(self, pred_ent_list, gold_ent_list, ere_cpg_dict, pattern):
        """
        ere_cpg_dict = {
                "ent_cpg": [0, 0, 0],
                }
        pattern: metric pattern
        """
        gold_ent_set = self.get_mark_sets_ent(gold_ent_list, pattern)
        pred_ent_set = self.get_mark_sets_ent(pred_ent_list, pattern)

        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent_cpg"])

    def get_cpg(
        self,
        sample_list,
        tok2char_span_list,
        batch_pred_shaking_tag,
        pattern="whole_text",
    ):
        """
        return correct number, predict number, gold number (cpg)
        """

        ere_cpg_dict = {
            "ent_cpg": [0, 0, 0],
        }

        # go through all sentences
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]

            pred_ent_list = self.shaking_tagger.decode_ent(
                text, pred_shaking_tag, tok2char_span
            )  # decoding
            gold_ent_list = sample["entity_list"]

            self.cal_ent_cpg(pred_ent_list, gold_ent_list, ere_cpg_dict, pattern)

            return ere_cpg_dict

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-12
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1
