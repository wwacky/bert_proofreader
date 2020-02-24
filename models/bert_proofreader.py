import logging

import torch
from transformers import BertForMaskedLM, BertJapaneseTokenizer


logger = logging.getLogger(__name__)


class BertProofreader:
    def __init__(self, pretrained_model: str,
                 cache_dir: str = None):

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)

        # Load pre-trained model (weights)
        self.model = BertForMaskedLM.from_pretrained(pretrained_model, cache_dir=cache_dir)
        self.model.to('cuda')

        # TODO: 場所変える
        self.model.eval()

    def mask_prediction(self, sentence: str) -> torch.Tensor:
        # 特殊Tokenの追加
        sentence = f'[CLS]{sentence}[SEP]'

        tokenized_text = self.tokenizer.tokenize(sentence)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens], device='cuda')

        # [MASK]に対応するindexを取得
        mask_index = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        # mask作成
        repeat_num = tokens_tensor.shape[1] - 2
        tokens_tensor = tokens_tensor.repeat(repeat_num, 1)
        # TODO: forループをやめてmask作って一括変更する
        for i in range(repeat_num):
            tokens_tensor[i, i + 1] = mask_index

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=None)
            predictions = outputs[0]

        return tokenized_text, predictions


    # def judge(self, tokens: List[str]):
    #     judges = []
    #     # TODO: forループをやめる
    #     for i, token in enumerate(tokens):
    #         pred_top_k_word = self.tokenizer.convert_ids_to_tokens(pred_top_k[i][i + 1])
    #         judges.append(token in pred_top_k_word)
    #         logger.info(f'{token}: {judges[-1]}')
    #         logger.debug(f'top k predicted word={pred_top_k_word}')
    #
    #     return all(judges)
    #
    def check_topk(self, sentence: str, topk: int = 10):

        tokens, predictions = self.mask_prediction(sentence)

        pred_sort = torch.argsort(predictions, dim=2, descending=True)
        pred_top_k = pred_sort[:, :, :topk]  # 上位Xのindex取得

        judges = []
        # TODO: forループをやめる
        for i in range(len(tokens) - 2):
            pred_top_k_word = self.tokenizer.convert_ids_to_tokens(pred_top_k[i][i + 1])
            judges.append(tokens[i + 1] in pred_top_k_word)
            logger.info(f'{tokens[i + 1]}: {judges[-1]}')
            logger.debug(f'top k word={pred_top_k_word}')

        return all(judges)

    def check_threshold(self, sentence: str, threshold: float = 0.01):
        tokens, predictions = self.mask_prediction(sentence)

        predictions = predictions.softmax(dim=2)

        judges = []
        # TODO: forループをやめる
        for i in range(len(tokens) - 2):
            indices = (predictions[i][i + 1] >= threshold).nonzero()
            pred_top_word = self.tokenizer.convert_ids_to_tokens(indices)
            judges.append(tokens[i + 1] in pred_top_word)
            logger.info(f'{tokens[i + 1]}: {judges[-1]}')
            logger.debug(f'word over threshold={pred_top_word}')

        return all(judges)
