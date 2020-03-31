from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm
import random

from covidfaq.rerank.predict import load_model


def make_qa_pairs(faq_path, n_wrong_answers=2, seed=42):
    with open(faq_path, 'r') as fh:
        faq  = json.load(fh)

    random.seed(seed)
    all_questions = []
    all_answers = []

    for k,v in faq.items():
       if k != 'document_URL':
           all_questions.append(k)
           all_answers.append("".join(faq[k]['plaintext']))

    qa_pairs = []
    for idx, question in enumerate(all_questions):
        correct_answer = all_answers[idx]
        wrong_answers = all_answers.copy()
        wrong_answers.remove(correct_answer)
        random.shuffle(wrong_answers)

        candidate_answers = []
        candidate_answers.append(correct_answer)
        candidate_answers.extend(wrong_answers[:n_wrong_answers])
        qa_pairs.append([question, candidate_answers])

    return qa_pairs

if __name__ == '__main__':

    model_name = 'bert-base-uncased'
    faq_path = 'covidfaq/scrape/quebec-en-faq.json'
    n_wrong_answers = 2 # number of wrong answers added to the correct answer

    bert_question = BertModel.from_pretrained(model_name)
    bert_paragraph = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = load_model(tokenizer, bert_question, bert_paragraph)

    accs = []
    # Run the test on 10 separate splits of the FAQ and average the results
    for seed in range(10):
        qa_pairs = make_qa_pairs(faq_path, n_wrong_answers=n_wrong_answers, seed=seed)
        correct = 0
        for question, answers in tqdm(qa_pairs):

            out = model.retriever.predict(question, answers)

            if out[2][0] == 0: # answers[0] is always the correct answer
                correct += 1

        acc = correct / len(qa_pairs) * 100
        accs.append(acc)
        print("single run accuracy: %", acc)

    print("Average model accuracy: ", sum(accs) / len(accs))
