import json
import os


def get_faq_files(scrape_path="covidfaq/scrape/"):
    jsonfiles = [
        f
        for f in os.listdir(scrape_path)
        if os.path.isfile(os.path.join(scrape_path, f))
        if ".json" in f
    ]
    enfiles = [scrape_path + f for f in jsonfiles if "quebec-en-faq" in f]
    frfiles = [scrape_path + f for f in jsonfiles if "quebec-fr-faq" in f]

    return enfiles, frfiles


def get_intents_from_json_files(files):
    doc_url_key = "document_URL"

    intents = []
    for i, file_ in enumerate(files):
        with open(file_, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        file_id = "en_" + str(i)
        question_id_ = 0
        for sec in json_file:
            if sec == doc_url_key:
                continue
            rec = {
                "question_id": file_id + "_q" + str(question_id_),
                "question": sec,
                "answer": json_file[sec]["plaintext"],
                "file_name": file_,
                "url": json_file[sec]["url"],
            }

            question_id_ += 1
            intents.append(rec)

    return intents


def create_nlu_intents_file(md_file, intents):
    if os.path.exists(md_file):
        os.remove(md_file)
    with open(md_file, "a+") as file_object:
        for intent in intents:
            file_object.write("## intent: ask_faq/ask_" + intent.get("question_id"))
            file_object.write("\n")
            file_object.write("- " + intent.get("question"))
            file_object.write("\n")
            file_object.write("\n")
        file_object.close()


def create_responses_file(md_file, intents):
    if os.path.exists(md_file):
        os.remove(md_file)
    with open(md_file, "a+") as file_object:
        for intent in intents:
            file_object.write("## " + intent.get("question_id"))
            file_object.write("\n")
            file_object.write("* ask_faq/ask_" + intent.get("question_id"))
            file_object.write("\n")
            file_object.write("  - " + " ".join(intent.get("answer")))
            file_object.write("\n")
            file_object.write("\n")
        file_object.close()


if __name__ == "__main__":

    md_path = "covidfaq/data/"

    enfiles, frfiles = get_faq_files()

    intents_en = get_intents_from_json_files(enfiles)
    intents_fr = get_intents_from_json_files(frfiles)

    create_nlu_intents_file(md_path + "en/nlu.md", intents_en)
    create_nlu_intents_file(md_path + "fr/nlu.md", intents_fr)

    create_responses_file(md_path + "en/responses.md", intents_en)
    create_responses_file(md_path + "fr/responses.md", intents_fr)
