import sys

sys.path.append("../")
import os
import json
from transformers import BertTokenizerFast
from common.utils import Preprocessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

config = {
    "encoder": "BERT",
    "bert_path": "../pretrained_models/bert-base-chinese",
}


def handle_normal_dataset(dataset, ignore_subword_match=False):
    """
    if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
    """
    # 加载preprocessor
    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(
            config["bert_path"], add_special_tokens=False, do_lower_case=False
        )
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )["offset_mapping"]
    elif config["encoder"] == "BiLSTM":
        tokenize = lambda text: text.split(" ")

        def get_tok2char_span_map(text):
            tokens = tokenize(text)
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    preprocessor = Preprocessor(
        tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map
    )
    # add char span
    dataset, miss_sample_list = preprocessor.add_char_span(
        dataset, ignore_subword_match=False
    )

    if len(miss_sample_list) > 0:
        print("=========存在不匹配实体，请检查===========")
        print(miss_sample_list)
        print("========================================")

    # add token span
    dataset = preprocessor.add_tok_span(dataset)

    return dataset



def main(data_path, data_type, out_dir, split_train_valid=False, split_radio=0.8, shuffle=True):
    """
    先将数据集先整理为预处理数据集的指定格式：
    [
        {
            "text":"2008年奥运会在北京举行。",
            "entity_list":[
                {"text":"2008","type":"year"},
                {"text":"北京","type":"loc"}
                ]
        }
    ]
    """

    cnt = 0
    ent_type_set = set()
    new_datas = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = {}
            data = json.loads(line)
            text = data["text"]
            
            item["id"] = cnt
            item["text"] = text
            cnt += 1
            if "test" in data_type:
                new_datas.append(item)
                continue
            
            labels = data["label"]
            item["entity_list"] = []
            for ent_type in labels:
                ent_type_set.add(ent_type)
                for ent_text in labels[ent_type]:
                    item["entity_list"].append({"text": ent_text, "type": ent_type})
            new_datas.append(item)
    
    out_path = os.path.join(out_dir, data_type + ".json")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "test" in data_type:
        json.dump(new_datas, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        return

    elif "train" in data_type:
        # 添加char_span,token_span成最终数据集
        dataset = handle_normal_dataset(new_datas, ignore_subword_match=False)
        if not split_train_valid:
            json.dump(dataset, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        else:
            train_dataset, valid_dataset = train_test_split(dataset, train_size=split_radio, shuffle=shuffle)
            json.dump(train_dataset, open(os.path.join(out_dir, "train_data.json"), "w", encoding="utf-8"), ensure_ascii=False)
            json.dump(valid_dataset, open(os.path.join(out_dir, "valid_data.json"), "w", encoding="utf-8"), ensure_ascii=False)

        # 保存实体类型信息
        ent_type_list = sorted(ent_type_set)
        ent2id = {ent: ind for ind, ent in enumerate(ent_type_list)}
        json.dump(
            ent2id, open(os.path.join(out_dir, "ent2id.json"), "w", encoding="utf-8"), ensure_ascii=False
        )
    else:
        # 添加char_span,token_span成最终数据集
        dataset = handle_normal_dataset(new_datas, ignore_subword_match=False)
        json.dump(dataset, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)



if __name__ == "__main__":
    out_dir = "../data4bert/cluener_new/"
    train_path = "./cluener_public/train.json"
    valid_path = "./cluener_public/dev.json"
    test_path = "./cluener_public/test.json"
    main(train_path, "train_data", out_dir)
    main(valid_path, "valid_data", out_dir)
    main(test_path, "test_data", out_dir)

