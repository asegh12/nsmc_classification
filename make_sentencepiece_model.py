from tqdm import tqdm
import re
import sentencepiece as spm

def make_naver_txt(filename):
    reviews = []
    with open(filename, "r", encoding="utf8") as f:
        with open('naver.txt', 'w', encoding='utf8') as f2:
            lines = f.readlines()
            for line in lines[1:]:
                tmp = '_' + re.sub(r"[^가-힣 ]", "", line.split('\t')[1])
                append_line = re.sub(r" ", "_", tmp)
                f2.write(f'{append_line}\n')
            
make_naver_txt("ratings_train.txt")
spm.SentencePieceTrainer.Train('--input=naver.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

