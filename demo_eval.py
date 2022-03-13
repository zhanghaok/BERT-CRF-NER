from logger import logger
from datasets import tqdm
import torch
from transformers import BertTokenizer
from model import BertForNER
from load_data import valdataloader,idx2label

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
device = "cuda:1" if torch.cuda.is_available() else 'cpu'
model = BertForNER.from_pretrained('./saved_model')
model.to(device)
model.eval()

def extract(chars,tags):
    """
    chars：一句话 "CLS  张    三   是我们  班    主   任   SEP"
    tags：标签列表[O   B-LOC,I-LOC,O,O,O,B-PER,I-PER,i-PER,O]
    返回一段话中的实体
    """
    result = []
    pre = ''
    w = []
    for idx,tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1] #pre LOC
                w.append(chars[idx])#w 张
        else:
            if tag == f'I-{pre}': #I-LOC True
                w.append(chars[idx]) #w 张三
            else:
                result.append([w,pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
    return [[''.join(x[0]),x[1]] for x in result]

gold_num = 0
predict_num = 0
correct_num = 0

pbar = tqdm(valdataloader)
for index,batch_data in enumerate(pbar):
    input_ids = batch_data['input_ids'].to(device)
    attention_mask = batch_data['attention_mask'].to(device)
    label_ids = batch_data['label_ids'].to(device)
    #验证集中真是的实体
    chars = tokenizer.convert_ids_to_tokens(input_ids[0])#由于valdatset的bathch_size = 1
    sent = ''.join(chars)
    logger.info('Sent: %s'%sent)
    label_ids[0][0]
    labels = [idx2label[ix.item()] for ix in label_ids[0]] #ix是个tensor(1)张量，ix.item()变为cpu类型的1
    entities = extract(chars,labels)#真实的实体返回列表
    gold_num += len(entities)
    logger.info('NER:%s'%entities)

    res = model(input_ids,attention_mask)
    pred_labels = [idx2label[ix] for ix in res[1]]
    pred_entities = extract(chars,pred_labels)

    predict_num += len(pred_entities)
    logger.info("Predict NER:%s"%pred_entities)
    logger.info("--------------------\n")

    for pred in pred_entities:
        if pred in entities:
            correct_num += 1

logger.info('gold_num = %d'%gold_num)
logger.info('predict_num = %d'%predict_num)
logger.info('correct_num = %d'%correct_num)
P = correct_num / predict_num
logger.info('P = %.4f'%P)
R = correct_num/gold_num
logger.info('R = %.4f'%R)
F1 = 2*P*R/(P+R)
logger.info('F1 = %.4f'%F1)


