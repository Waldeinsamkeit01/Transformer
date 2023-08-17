# -*- coding:utf-8 -*- 
# coding:unicode_escape
# @Author: Lemon00
# @Time: 2023/8/12 14:37
# @File: piece
import sentencepiece as spm
import jieba


# spm.SentencePieceTrainer.train('--input=../example_10M.csv --model_prefix=model-t --vocab_size=65000')
# sp = spm.SentencePieceProcessor(model_file='model-t-en.model')
# test = sp.encode('')
# print(test)

# 导入分词器
# def load_tokenizers():
#     try:
#         # 英文分词器选用自己的sentencePiece
#         tokenizer_en = spm.SentencePieceProcessor(model_file='model-t-en.model')
#     except IOError:
#         print("Tokenizer_en can't load")
#     try:
#         # 中文分词器选用jieba分词
#         tokenizer_zh = jieba.Tokenizer()
#     except IOError:
#         print("Tokenizer_zh can't load")
#     return tokenizer_en, tokenizer_zh

# 导入分词器
def load_tokenizers():
    try:
        # 英文分词器
        tokenizer_en = spm.SentencePieceProcessor(model_file='model-t-en.model')
    except IOError:
        print("Tokenizer_en can't load")
    try:
        # 中文分词器
        tokenizer_zh = spm.SentencePieceProcessor(model_file='model-t-zh.model')
    except IOError:
        print("Tokenizer_zh can't load")
    return tokenizer_en, tokenizer_zh



# tokenizer_en, tokenizer_zh = load_tokenizers()
# t = tokenizer_zh.cut("我喜欢结巴分词")
# l = list(t)
# print(' '.join(t))
# print(l)


# 进行分词
def tokenize(text, tokenizer):
    return [tok for tok in tokenizer.Encode(text)]

tokenizer_en, tokenizer_zh = load_tokenizers()
t_en = tokenize("I love China", tokenizer_en)
t_zh = tokenize("我爱中国", tokenizer_zh)
print(t_en, t_zh)
print(len(tokenizer_en))
print(tokenizer_en.pad_id())
# print(tokenizer_en.Encode(t_en))


# TODO: 看不懂
def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])



# def build_vocabulary(spacy_de, spacy_en):
#     def tokenize_de(text):
#         return tokenize(text, spacy_de)
#
#     def tokenize_en(text):
#         return tokenize(text, spacy_en)
#
#     print("Building German Vocabulary ...")
#     train, val, test = datasets.Multi30k(language_pair=("de", "en"))
#     vocab_src = build_vocab_from_iterator(
#         yield_tokens(train + val + test, tokenize_de, index=0),
#         min_freq=2,
#         specials=["<s>", "</s>", "<blank>", "<unk>"],
#     )
#
#     print("Building English Vocabulary ...")
#     train, val, test = datasets.Multi30k(language_pair=("de", "en"))
#     vocab_tgt = build_vocab_from_iterator(
#         yield_tokens(train + val + test, tokenize_en, index=1),
#         min_freq=2,
#         specials=["<s>", "</s>", "<blank>", "<unk>"],
#     )
#
#     vocab_src.set_default_index(vocab_src["<unk>"])
#     vocab_tgt.set_default_index(vocab_tgt["<unk>"])
#
#     return vocab_src, vocab_tgt
#
#
# def load_vocab(spacy_de, spacy_en):
#     if not exists("vocab.pt"):
#         vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
#         torch.save((vocab_src, vocab_tgt), "vocab.pt")
#     else:
#         vocab_src, vocab_tgt = torch.load("vocab.pt")
#     print("Finished.\nVocabulary sizes:")
#     print(len(vocab_src))
#     print(len(vocab_tgt))
#     return vocab_src, vocab_tgt
#
#
# if is_interactive_notebook():
#     # global variables used later in the script
#     spacy_de, spacy_en = show_example(load_tokenizers)
#     vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])



