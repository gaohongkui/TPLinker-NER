import sys

sys.path.append("../")

import config
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast
from torch.utils.data import DataLoader, Dataset
from common.utils import Preprocessor
import json
from tqdm import tqdm
import glob
from glove import Glove
import re
import copy
from tplinker_plus_ner import (
    HandshakingTaggingScheme,
    DataMaker4Bert,
    DataMaker4BiLSTM,
    TPLinkerPlusBert,
    TPLinkerPlusBiLSTM,
    MetricsCalculator,
)
from pprint import pprint

config = config.eval_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0  # num_workers在windows系统下有bug

data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = config["hyper_parameters"]["batch_size"]
ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
max_seq_len = config["hyper_parameters"]["max_seq_len"]
# for reproductivity
torch.backends.cudnn.deterministic = True

if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(
        config["bert_path"], add_special_tokens=False, do_lower_case=False
    )
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(
        text, return_offsets_mapping=True, add_special_tokens=False
    )["offset_mapping"]
elif config["encoder"] in {"BiLSTM", }:
    tokenize = lambda text: text.split(" ")


    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

preprocessor = Preprocessor(
    tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map
)

# Load Data
test_data_path_dict = {}
for file_path in glob.glob(test_data_path):
    file_name = re.search("(.*?)\.json", file_path.split(os.sep)[-1]).group(1)
    test_data_path_dict[file_name] = file_path

test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))

all_data = []
for data in list(test_data_dict.values()):
    all_data.extend(data)

max_tok_num = 0
for sample in tqdm(all_data, desc="Calculate the max token number"):
    tokens = tokenize(sample["text"])
    max_tok_num = max(len(tokens), max_tok_num)

# Split Data
split_test_data = False
if max_tok_num > max_seq_len:
    split_test_data = True
    print(
        "max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(
            max_tok_num, max_seq_len
        )
    )
else:
    print(
        "max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(
            max_tok_num, max_seq_len
        )
    )
max_seq_len = min(max_tok_num, max_seq_len)

if config["hyper_parameters"]["force_split"]:
    split_test_data = True
    print("force to split the test dataset!")

ori_test_data_dict = copy.deepcopy(test_data_dict)
if split_test_data:
    test_data_dict = {}
    for file_name, data in ori_test_data_dict.items():
        test_data_dict[file_name] = preprocessor.split_into_short_samples(
            data,
            max_seq_len,
            sliding_len=config["hyper_parameters"]["sliding_len"],
            encoder=config["encoder"],
            data_type="test",
        )

ent2id = json.load(open(ent2id_path, "r", encoding="utf-8"))
handshaking_tagger = HandshakingTaggingScheme(ent2id, max_seq_len)
tag_size = handshaking_tagger.get_tag_size()

# Data Maker
if config["encoder"] == "BERT":
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)

elif config["encoder"] in {
    "BiLSTM",
}:
    token2idx_path = os.path.join(*config["token2idx_path"])
    token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
    idx2token = {idx: tok for tok, idx in token2idx.items()}


    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx["<UNK>"])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx["<PAD>"]] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids


    data_maker = DataMaker4BiLSTM(
        text2indices, get_tok2char_span_map, handshaking_tagger
    )


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Model
if config["encoder"] == "BERT":
    encoder = BertModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    ent_extractor = TPLinkerPlusBert(
        encoder,
        tag_size,
        config["hyper_parameters"]["shaking_type"],
        config["hyper_parameters"]["inner_enc_type"],
    )

elif config["encoder"] in {"BiLSTM", }:
    glove = Glove()
    glove = glove.load(config["pretrained_word_embedding_path"])

    # prepare embedding matrix
    word_embedding_init_matrix = np.random.normal(
        -1, 1, size=(len(token2idx), config["word_embedding_dim"])
    )
    count_in = 0

    # 在预训练词向量中用该预训练向量,不使用随机向量
    for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
        if tok in glove.dictionary:
            count_in += 1
            word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]

    print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token)))  # 命中预训练词向量的比例
    word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

    ent_extractor = TPLinkerPlusBiLSTM(
        word_embedding_init_matrix,
        config["emb_dropout"],
        config["enc_hidden_size"],
        config["dec_hidden_size"],
        config["rnn_dropout"],
        tag_size,
        config["hyper_parameters"]["shaking_type"],
        config["hyper_parameters"]["inner_enc_type"],
    )

ent_extractor = ent_extractor.to(device)

metrics = MetricsCalculator(handshaking_tagger)

# Load trained model path
model_state_dir = config["model_state_dict_dir"]
target_run_ids = set(config["run_ids"])
run_id2model_state_paths = {}
for root, dirs, files in os.walk(model_state_dir):
    for file_name in files:
        run_id = root.split(os.sep)[-2].split("-")[-1] if model_state_dir == "./wandb" else \
        root.split(os.sep)[-1].split("-")[
            -1]
        if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
            if run_id not in run_id2model_state_paths:
                run_id2model_state_paths[run_id] = []
            model_state_path = os.path.join(root, file_name)
            run_id2model_state_paths[run_id].append(model_state_path)

assert len(run_id2model_state_paths) != 0, "未加载到已训练模型，请检查路径及run_id"


def get_last_k_paths(path_list, k):
    path_list = sorted(
        path_list, key=lambda x: int(re.search("(\d+)", x.split(os.sep)[-1]).group(1))
    )
    return path_list[-k:]


# only last k models
k = config["last_k_model"]
for run_id, path_list in run_id2model_state_paths.items():
    run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)


def filter_duplicates(ent_list):
    """
    过滤重复实体
    """
    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(
            ent["tok_span"][0], ent["tok_span"][1], ent["type"]
        )
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)

    return filtered_ent_list


def predict(test_data, ori_test_data):
    """
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    """
    indexed_test_data = data_maker.get_indexed_data(
        test_data, max_seq_len, data_type="test"
    )  # fill up to max_seq_len
    test_dataloader = DataLoader(
        MyDataset(indexed_test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
        collate_fn=lambda data_batch: data_maker.generate_batch(
            data_batch, data_type="test"
        ),
    )

    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc="Predicting"):
        if config["encoder"] == "BERT":
            (
                sample_list,
                batch_input_ids,
                batch_attention_mask,
                batch_token_type_ids,
                tok2char_span_list,
                _,
            ) = batch_test_data

            batch_input_ids, batch_attention_mask, batch_token_type_ids = (
                batch_input_ids.to(device),
                batch_attention_mask.to(device),
                batch_token_type_ids.to(device),
            )

        elif config["encoder"] in {
            "BiLSTM",
        }:
            sample_list, batch_input_ids, tok2char_span_list, _ = batch_test_data

            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] == "BERT":
                batch_pred_shaking_tag, _ = ent_extractor(
                    batch_input_ids, batch_attention_mask, batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                batch_pred_shaking_tag, _ = ent_extractor(batch_input_ids)

        batch_pred_shaking_tag = (batch_pred_shaking_tag > 0.0).long()

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]
            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = (
                    gold_sample["tok_offset"],
                    gold_sample["char_offset"],
                )
            ent_list = handshaking_tagger.decode_ent(
                text,
                pred_shaking_tag,
                tok2char_span,
                tok_offset=tok_offset,
                char_offset=char_offset,
            )
            pred_sample_list.append(
                {"text": text, "id": text_id, "entity_list": ent_list, }
            )

    # merge
    text_id2pred_res = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2pred_res:
            text_id2pred_res[text_id] = {
                "ent_list": sample["entity_list"],
            }
        else:
            text_id2pred_res[text_id]["ent_list"].extend(sample["entity_list"])

    text_id2text = {sample["id"]: sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, pred_res in text_id2pred_res.items():
        filtered_ent_list = filter_duplicates(pred_res["ent_list"])
        merged_pred_sample_list.append(
            {
                "id": text_id,
                "text": text_id2text[text_id],
                "entity_list": filtered_ent_list,
            }
        )

    return merged_pred_sample_list


# predict
res_dict = {}
predict_statistics = {}
for file_name, short_data in test_data_dict.items():
    ori_test_data = ori_test_data_dict[file_name]
    for run_id, model_path_list in run_id2model_state_paths.items():
        save_dir4run = os.path.join(save_res_dir, run_id)
        if config["save_res"] and not os.path.exists(save_dir4run):
            os.makedirs(save_dir4run)

        for model_state_path in model_path_list:
            res_num = re.search("(\d+)", model_state_path.split(os.sep)[-1]).group(1)
            save_path = os.path.join(
                save_dir4run, "{}_res_{}.json".format(file_name, res_num)
            )

            if os.path.exists(save_path):
                pred_sample_list = [
                    json.loads(line) for line in open(save_path, "r", encoding="utf-8")
                ]
                print("{} already exists, load it directly!".format(save_path))
            else:
                # load model state
                ent_extractor.load_state_dict(torch.load(model_state_path))
                ent_extractor.eval()
                print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split(os.sep)[-1]))

                # predict
                pred_sample_list = predict(short_data, ori_test_data)

            res_dict[save_path] = pred_sample_list
            predict_statistics[save_path] = len(
                [s for s in pred_sample_list if len(s["entity_list"]) > 0]
            )
pprint(predict_statistics)

# check
for path, res in res_dict.items():
    for sample in tqdm(res, desc="check char span"):
        text = sample["text"]
        for ent in sample["entity_list"]:
            assert ent["text"] == text[ent["char_span"][0]: ent["char_span"][1]]

# save
if config["save_res"]:
    for path, res in res_dict.items():
        with open(path, "w", encoding="utf-8") as file_out:
            for sample in tqdm(res, desc="Output"):
                if len(sample["entity_list"]) == 0:
                    continue
                json_line = json.dumps(sample, ensure_ascii=False)
                file_out.write("{}\n".format(json_line))


# score
def get_test_prf(pred_sample_list, gold_test_data, pattern="whole_text"):
    """
    测试集Precision,Recall,F1
    要求测试集必须已标注（按验证集格式标注）
    """
    text_id2gold_n_pred = {}  # text id to gold and pred results

    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_entity_list": sample["entity_list"],
        }

    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

    ere_cpg_dict = {
        "ent_cpg": [0, 0, 0],
    }
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_ent_list = gold_n_pred["gold_entity_list"]
        pred_ent_list = (
            gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []
        )
        metrics.cal_ent_cpg(pred_ent_list, gold_ent_list, ere_cpg_dict, pattern)

    ent_prf = metrics.get_prf_scores(
        ere_cpg_dict["ent_cpg"][0],
        ere_cpg_dict["ent_cpg"][1],
        ere_cpg_dict["ent_cpg"][2],
    )
    prf_dict = {
        "ent_prf": ent_prf,
    }
    return prf_dict


if config["score"]:
    filepath2scores = {}
    for file_path, pred_samples in res_dict.items():
        file_name = re.search("(.*?)_res_\d+\.json", file_path.split(os.sep)[-1]).group(1)
        gold_test_data = ori_test_data_dict[file_name]
        prf_dict = get_test_prf(
            pred_samples,
            gold_test_data,
            pattern=config["hyper_parameters"]["match_pattern"],
        )
        filepath2scores[file_path] = prf_dict
    print("---------------- Results -----------------------")
    pprint(filepath2scores)
