
import pandas as pd
import regex, re

import spacy
from spacy import displacy

def fuzzy_match(word,pattern):
    '''
        Fuzzy matching function to be used with .apply() of pandas

        Reason - Fuzzy matching is available in regex package, not in re package, 
        therefore fuzzy matching is not a part of pandas string matching functions
    '''
 
    if regex.search(pattern, word, re.IGNORECASE):
        return True
    else:
        return False  

def save_dep_graph_html(df):
    i=0 
    svg = ""
    def save_dep_svg(doc):
        global svg
        svg += "<br/><br/>" + spacy.displacy.render(doc, style="dep", jupyter=False, options={"add_lemma":True})


    df.apply(save_dep_svg)


    html = '''<!DOCTYPE html>
                <html>
                <body>'''\
                + svg +\
                '''
                </body>
                </html>
                '''

    output_path = os.path.join("./output/dependency_relations/", f"dependencies.html")
    with open(output_path, "w+") as file:
        file.write(html)

def load_spacy_nlp():

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp_en = spacy.load("en_core_web_sm")
    # nlp_fr = spacy.load("fr_core_news_md")

    return nlp_en

# Let's get all the tuples for the whole corpus
def get_tuples(doc):  
    nlp_en = load_spacy_nlp()

    tuples = []
    if type(doc) == str:
        doc = nlp_en(doc)
    for sentence in doc.sents:  
        for token in sentence:            
            # Get SV tuples
            if "nsubj" in token.dep_:                  
                for sibling in token.head.children:
                    if sibling.dep_ in ["dobj","acomp"]:
                        tuples.append((token.text.lower(), token.head.text.lower(), sibling.text.lower(), token.lemma_.lower(), token.head.lemma_.lower(), sibling.lemma_.lower(), sibling.dep_))  
    return tuples

# Let's get all the dependencies for the whole corpus
def get_dependencies(doc):  
    nlp_en = load_spacy_nlp()

    dependencies = []
    if type(doc) == str:
        doc = nlp_en(doc)
    for sentence in doc.sents:  
        for token in sentence:            
            dependencies.append((token.lemma_.lower(), token.dep_, token.head.lemma_.lower()))  
    return dependencies

from scipy.sparse import coo_matrix
def get_vectors(rows, features, binary=True):
    if binary:
        val=1
    mtxr, mtxc, mtxv = [], [], [] 
    for i,row in enumerate(rows):
        sequence = [features.index(feat_val) for feat_val in row if feat_val in features]
        mtxr.extend([i]*len(sequence))
        mtxv.extend([1]*len(sequence))
        mtxc.extend([j for j in sequence])
    return coo_matrix((mtxv, (mtxr, mtxc)), shape=(len(rows), len(features)))

def interpret_vectors(vectors, features):
    if np.array(vectors).shape[-1:][0] != len(features):
        raise Exception("Vector size does not match number of features")  
    
    if len(np.array(vectors).shape) == 1:      
        vector = vectors # single vector 
        return [feat for val,feat in zip(vector, features) if val>0]
    else:        
        return [[feat for val,feat in zip(vector, features) if val>0] for vector in vectors]       

def get_dep_tokens(doc, dep, head=True, lemma=True):
    if head:
        return list(set([token.head.lemma_.lower() if lemma else token.head.text.lower() for sent in doc.sents for token in sent if dep.lower() in token.dep_.lower()]))    
    else:
        return list(set([token.lemma_.lower() if lemma else token.text.lower() for sent in doc.sents for token in sent if dep.lower() in token.dep_.lower()]))    

def prepare_dependencies(dataset):
    dataset["tuples"] = dataset.spacy_doc.apply(get_tuples)
    dataset["dependencies"] = dataset.spacy_doc.apply(get_dependencies)
    dataset["nsubj_lemma"] = dataset.spacy_doc.apply(get_dep_tokens, dep="nsubj", lemma=True)
    dataset["nsubj_token"] = dataset.spacy_doc.apply(get_dep_tokens, dep="nsubj", lemma=False)
    dataset["nsubj_dep_lemma"] = dataset.spacy_doc.apply(get_dep_tokens, dep="nsubj", head=False, lemma=True)
    dataset["nsubj_dep_token"] = dataset.spacy_doc.apply(get_dep_tokens, dep="nsubj", head=False, lemma=False)
    dataset["root_lemma"] = dataset.spacy_doc.apply(get_dep_tokens, dep="root", lemma=True)
    dataset["root_token"] = dataset.spacy_doc.apply(get_dep_tokens, dep="root", lemma=False)
    return dataset

def generate_dep_vectors(dataset):
    # to be called after prepare_dependencies method

    # Dataframe with all the unique tuples
    tuples = list(set([t for row in dataset["tuples"].tolist() for t in row]))

    # Dataframe with all the unique dependencies
    # dependencies = pd.DataFrame([t for row in dataset["dependencies"].tolist() for t in row], columns=["dep","rel","head"])
    # dependencies = dependencies.drop_duplicates()
    dependencies = list(set([t for row in dataset["dependencies"].tolist() for t in row]))

    dataset["dependency_vector"] = list(get_vectors(dataset.dependencies, dependencies).toarray())

    dataset["tuple_vector"] = list(get_vectors(dataset.tuples, tuples).toarray())

    return dataset

def val_isin(search_in, search, verbose=False):
    if type(search) != list:
        search = [search]
    # If tuple to match against
    boolean_list =[]
    for search_item in search:
        if type(search_item) == tuple:
            for item in search_in:
                match = True
                if verbose: print(list(zip(search_item, item)))
                for pattern, piece in zip(search_item, item):                    
                    if type(pattern) == list:
                        if piece not in pattern:
                            match = False
                            if verbose: print(pattern, piece, match, sep="\t")
                            break
                        else:                            
                            if verbose: print(pattern, piece, match, sep="\t")
                    elif pattern in ["*","",None]:
                        if verbose: print(pattern, piece, match, sep="\t")
                        continue
                    elif pattern is not piece:
                        match = False
                        if verbose: print(pattern, piece, match, sep="\t")
                        break
                    else:                        
                        if verbose: print(pattern, piece, match, sep="\t")
                boolean_list.append(match) 
                if verbose: print()         
        else:
            boolean_list = [search_item in search_in]

    if verbose: print(boolean_list.count(True))
    # return any(item in search_in for item in search)
    return any(boolean_list)


def sort_into_clusters(dataset, text_col="text", cluster_col="cluster"):
    # Set default value
    dataset[cluster_col] = "unclassified"

    # # Separate very long questions out
    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (dataset.len > 15)
    #                 , cluster_col ] = "too-much-information"

    # Questions with first and second person pronoun dependent in nominal subject relation
    dataset.loc[
                    (dataset[cluster_col]=="unclassified") & 
                    (dataset.nsubj_dep_token.apply(val_isin, search=["i","we","you"]))
                , cluster_col ] = "personal"


    # Questions WITHOUT first and second person pronoun dependent in nominal subject relation
    dataset.loc[
                    (dataset[cluster_col]=="unclassified") & 
                    (~dataset.nsubj_dep_token.apply(val_isin, search=["i","we","you"]))
                , cluster_col ] = "covid"

    #1 Situation statistics - Token rules were already sufficient
    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (~dataset.nsubj_dep_token.apply(val_isin, search=["i","we","you"])) 
                        (
                            dataset[text_col].str.contains("cases",case=False)|
                            dataset[text_col].str.contains("dea(?:th|d)(?:ly)?",case=False)|
                            dataset[text_col].str.contains("died",case=False)|
                            dataset[text_col].str.contains("(?:mortality|death|fatality) rate",case=False)|
                            dataset[text_col].str.contains("statistic",case=False)|
                            (
                                dataset[text_col].str.contains("how",case=False)&
                                dataset[text_col].str.contains("many",case=False)&
                                dataset[text_col].str.contains("people",case=False)
                            )
                        )
                    , cluster_col ] = "situation-stats"

    #2 Transmission - Animals - Token rules were already sufficient
    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (~dataset.nsubj_dep_token.apply(val_isin, search=["i","we","you"])) 
                        (dataset[text_col].str.contains(r"\b(?:animal|bird|cat|dog|pet)s?\b",case=False))
                    , cluster_col ] = "covid-transmission-animals"

    #3 Precaution - Gear - Token rules were already sufficient (for now), better without dependency based filtering
    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        # (dataset.nsubj_dep_token.apply(val_isin, search=["i","we","you"])) 
                        (
                            dataset[text_col].str.contains("mask",case=False)|
                            dataset[text_col].str.contains("glove",case=False)
                        )
                    , cluster_col ] = "covid-precaution-gear"

    #3 Precaution - Disinfection - Token rules were already sufficient (for now), better without dependency based filtering
    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("wash",case=False)|
                            dataset[text_col].str.contains("disinfect",case=False)
                        )
                    , cluster_col ] = "covid-precaution-disinfection"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"\bisolat",case=False)|
                            dataset[text_col].str.contains(r"\bsocial dist",case=False)|
                            dataset[text_col].str.contains(r"\bconfine",case=False)
                        )
                    , cluster_col ] = "covid-precaution-isolation"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                (
                                    dataset[text_col].str.contains("go (?:on|to|for|out)",case=False)|
                                    dataset[text_col].str.contains("walk",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("allow",case=False)|
                                    dataset[text_col].str.contains("can",case=False)|
                                    dataset[text_col].str.contains("ok|okay",case=False)|
                                    dataset[text_col].str.contains("should|shall",case=False)
                                )
                            )|
                            (
                                dataset[text_col].str.contains("lockdown",case=False)|
                                dataset[text_col].str.contains(r"\bopen\b",case=False)|
                                dataset[text_col].str.contains(r"\bclose",case=False)
                            )
                        )
                    , cluster_col ] = "situation-lockdown"

    # Questions with first and second person pronoun dependent in nominal subject relation
    dataset.loc[
                    (dataset[cluster_col]=="unclassified") & 
                    (dataset.tuples.apply(val_isin, search=(["i","we","you"],"do","what","-pron-",None,None,None)))
                , cluster_col ] = "personal-situation"


    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("if i (?:have|am|m)",case=False)
                        )
                    , cluster_col ] = "covid-whatif"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"i (?:have|ve)",case=False)|
                            (
                                dataset[text_col].str.contains(r"\b(?:has|have)",case=False)&
                                dataset[text_col].str.contains(r"symptom",case=False)
                            )
                            # dataset[text_col].str.contains(r"(?:i (?:think|feel) )?i \b(?:have|ve|am|m)\b",case=False)
                        )
                    , cluster_col ] = "personal-symptoms"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"\btransmi|contract|catch|spread|airborne",case=False)
                        )                
                    , cluster_col ] = "covid-transmission"

    dataset.loc[
                        (dataset[cluster_col]=="covid-transmission") & 
                        (
                            dataset[text_col].str.contains("again",case=False)|
                            dataset[text_col].str.contains("twice",case=False)
                        )                
                    , cluster_col ] = "covid-transmission-twice"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                                (
                                    dataset[text_col].str.contains("(?:corona|covid)",case=False)|
                                    dataset[text_col].str.contains("virus",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("live|stay|survive",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("on",case=False)
                                )
                        )
                    , cluster_col ] = "covid-life"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("infected",case=False)|
                            dataset[text_col].str.contains("infection",case=False)
                        )                
                    , cluster_col ] = "covid-infection"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                         dataset[text_col].str.contains("prevent",case=False)|
    #                         dataset[text_col].str.contains("protect",case=False)|
    #                         dataset[text_col].str.contains("precaution",case=False)|
    #                         dataset[text_col].str.contains("safety",case=False)|
    #                         (
    #                             dataset[text_col].str.contains("keep",case=False)&
    #                             dataset[text_col].str.contains("safe",case=False)
    #                         )
    #                     )               
    #                 , cluster_col ] = "covid-precaution"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                             (
    #                                 dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
    #                                 dataset[text_col].str.contains("corona",case=False)|
    #                                 dataset[text_col].str.contains("virus",case=False)
    #                             )&
    #                             (
    #                                 dataset[text_col].str.contains("kills",case=False)
    #                             )
    #                     )                
    #               , cluster_col ] = "covid-kill"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                             (
    #                                 (
    #                                     dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
    #                                     dataset[text_col].str.contains("corona",case=False)|
    #                                     dataset[text_col].str.contains("virus",case=False)
    #                                 )&
    #                                 (
    #                                     dataset[text_col].str.contains("fight",case=False)
    #                                 )&
    #                                 (
    #                                     dataset[text_col].str.contains("help",case=False)
    #                                 )
    #                             )|
    #                             (
    #                                 dataset[text_col].str.contains("mask",case=False)|
    #                                 dataset[text_col].str.contains("glove",case=False)
    #                             )
    #                         )                
    #                     , cluster_col ] = "covid-fight"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("treatment",case=False)|
                            dataset[text_col].str.contains("cure",case=False)|
                            dataset[text_col].str.contains("vaccine",case=False)|
                            dataset[text_col].str.contains("medic",case=False)
                        )                
                    , cluster_col ] = "covid-med"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("incubate",case=False)|
                            dataset[text_col].str.contains("incubation",case=False)
                        )      
                    , cluster_col ] = "covid-incubation"

    # dataset.loc[                     
    #                        (dataset[cluster_col]=="unclassified") & (
    # #                         dataset[text_col].str.contains(r"\bgo\b",case=False)&
    #                         (
    #                             dataset[text_col].str.contains("hospital",case=False)|                            
    #                             dataset[text_col].str.contains(r"\bER\b",case=False)
    #                         )
    #                     )
    #                , cluster_col ] = "hospital"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("diff",case=False)|
                            dataset[text_col].apply(fuzzy_match, pattern="(?:distinguish){e<=3}")
                        )                
                    , cluster_col ] = "covid-versus"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("tested",case=False)|
                            dataset[text_col].str.contains("test",case=False)
                        )                
                    , cluster_col ] = "personal-testing"

    dataset.loc[
                        (dataset[cluster_col]=="personal-testing") & 
                        (
                            dataset[text_col].str.contains("(?:tested|test)",case=False) &
                            dataset[text_col].str.contains("where",case=False)
                        )                
                    , cluster_col ] = "personal-testing-location"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("recover",case=False)
                        )                
                    , cluster_col ] = "covid-recovery"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("at risk",case=False)|
                            (
                                dataset[text_col].str.contains("more",case=False) &
                                dataset[text_col].str.contains("risky|dangerous",case=False)
                            )
                        )                
                    , cluster_col ] = "covid-vulnerable"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("dangerous",case=False)|
                            dataset[text_col].str.contains("risk",case=False)
                        )                
                    , cluster_col ] = "covid-contagious"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (dataset[text_col].str.contains(r"\bsymptom",case=False))                
                    , cluster_col ] = "covid-symptoms"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                dataset[text_col].apply(fuzzy_match, pattern="(?:whats|what (?:is|s))")
                            ) & 
                            (
                                dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
                                dataset[text_col].str.contains("corona",case=False)
                            )
                        )                
                    , cluster_col ] = "covid-what"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                (
                                    dataset[text_col].str.contains("how",case=False) &
                                    dataset[text_col].str.contains("long",case=False)
                                )|
                                dataset[text_col].str.contains("when",case=False)
                            )&
                                dataset[text_col].str.contains("will",case=False)&
                            (
                                dataset[text_col].str.contains("last|end|over|normal|done",case=False)
                            )
                        )                
                    , cluster_col ] = "situation-future"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                dataset[text_col].str.contains("how|when|where",case=False) 
                            )&
                                dataset[text_col].str.contains("did",case=False)&
                            (
                                dataset[text_col].str.contains("start|begin|began",case=False)
                            )
                        )                
                    , cluster_col ] = "situation-past"

    return dataset


def apply_token_rules(dataset, text_col="text", cluster_col="cluster"):
    # Set default value
    dataset[cluster_col] = "unclassified"

    # # Separate very long questions out
    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (dataset.len > 15)
    #                 , cluster_col ] = "too-much-information"

    # Statistics
    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("cases",case=False)|
                            dataset[text_col].str.contains("dea(?:th|d)(?:ly)?",case=False)|
                            dataset[text_col].str.contains("died",case=False)|
                            dataset[text_col].str.contains("(?:mortality|death|fatality) rate",case=False)|
                            dataset[text_col].str.contains("statistic",case=False)|
                            (
                                dataset[text_col].str.contains("how",case=False)&
                                dataset[text_col].str.contains("many",case=False)&
                                dataset[text_col].str.contains("people",case=False)
                            )
                        )
                    , cluster_col ] = "situation-stats"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (dataset[text_col].str.contains(r"\b(?:animal|bird|cat|dog|pet)s?\b",case=False))
                    , cluster_col ] = "covid-transmission-animals"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("mask",case=False)|
                            dataset[text_col].str.contains("glove",case=False)
                        )
                    , cluster_col ] = "covid-precaution-gear"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("wash",case=False)|
                            dataset[text_col].str.contains("disinfect",case=False)
                        )
                    , cluster_col ] = "covid-precaution-disinfection"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"\bisolat",case=False)|
                            dataset[text_col].str.contains(r"\bsocial dist",case=False)|
                            dataset[text_col].str.contains(r"\bconfine",case=False)
                        )
                    , cluster_col ] = "covid-precaution-isolation"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                (
                                    dataset[text_col].str.contains("go (?:on|to|for|out)",case=False)|
                                    dataset[text_col].str.contains("walk",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("allow",case=False)|
                                    dataset[text_col].str.contains("can",case=False)|
                                    dataset[text_col].str.contains("ok|okay",case=False)|
                                    dataset[text_col].str.contains("should|shall",case=False)
                                )
                            )|
                            (
                                dataset[text_col].str.contains("lockdown",case=False)|
                                dataset[text_col].str.contains(r"\bopen\b",case=False)|
                                dataset[text_col].str.contains(r"\bclose",case=False)
                            )
                        )
                    , cluster_col ] = "situation-lockdown"



    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("if i (?:have|am|m)",case=False)
                        )
                    , cluster_col ] = "covid-whatif"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"i (?:have|ve)",case=False)|
                            (
                                dataset[text_col].str.contains(r"\b(?:has|have)",case=False)&
                                dataset[text_col].str.contains(r"symptom",case=False)
                            )
                            # dataset[text_col].str.contains(r"(?:i (?:think|feel) )?i \b(?:have|ve|am|m)\b",case=False)
                        )
                    , cluster_col ] = "personal-symptoms"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains(r"\btransmi|contract|catch|spread|airborne",case=False)
                        )                
                    , cluster_col ] = "covid-transmission"

    dataset.loc[
                        (dataset[cluster_col]=="covid-transmission") & 
                        (
                            dataset[text_col].str.contains("again",case=False)|
                            dataset[text_col].str.contains("twice",case=False)
                        )                
                    , cluster_col ] = "covid-transmission-twice"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                                (
                                    dataset[text_col].str.contains("(?:corona|covid)",case=False)|
                                    dataset[text_col].str.contains("virus",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("live|stay|survive",case=False)
                                )&
                                (
                                    dataset[text_col].str.contains("on",case=False)
                                )
                        )
                    , cluster_col ] = "covid-life"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("infected",case=False)|
                            dataset[text_col].str.contains("infection",case=False)
                        )                
                    , cluster_col ] = "covid-infection"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                         dataset[text_col].str.contains("prevent",case=False)|
    #                         dataset[text_col].str.contains("protect",case=False)|
    #                         dataset[text_col].str.contains("precaution",case=False)|
    #                         dataset[text_col].str.contains("safety",case=False)|
    #                         (
    #                             dataset[text_col].str.contains("keep",case=False)&
    #                             dataset[text_col].str.contains("safe",case=False)
    #                         )
    #                     )               
    #                 , cluster_col ] = "covid-precaution"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                             (
    #                                 dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
    #                                 dataset[text_col].str.contains("corona",case=False)|
    #                                 dataset[text_col].str.contains("virus",case=False)
    #                             )&
    #                             (
    #                                 dataset[text_col].str.contains("kills",case=False)
    #                             )
    #                     )                
    #               , cluster_col ] = "covid-kill"

    # dataset.loc[
    #                     (dataset[cluster_col]=="unclassified") & 
    #                     (
    #                             (
    #                                 (
    #                                     dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
    #                                     dataset[text_col].str.contains("corona",case=False)|
    #                                     dataset[text_col].str.contains("virus",case=False)
    #                                 )&
    #                                 (
    #                                     dataset[text_col].str.contains("fight",case=False)
    #                                 )&
    #                                 (
    #                                     dataset[text_col].str.contains("help",case=False)
    #                                 )
    #                             )|
    #                             (
    #                                 dataset[text_col].str.contains("mask",case=False)|
    #                                 dataset[text_col].str.contains("glove",case=False)
    #                             )
    #                         )                
    #                     , cluster_col ] = "covid-fight"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("treatment",case=False)|
                            dataset[text_col].str.contains("cure",case=False)|
                            dataset[text_col].str.contains("vaccine",case=False)|
                            dataset[text_col].str.contains("medic",case=False)
                        )                
                    , cluster_col ] = "covid-med"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("incubate",case=False)|
                            dataset[text_col].str.contains("incubation",case=False)
                        )      
                    , cluster_col ] = "covid-incubation"

    # dataset.loc[                     
    #                        (dataset[cluster_col]=="unclassified") & (
    # #                         dataset[text_col].str.contains(r"\bgo\b",case=False)&
    #                         (
    #                             dataset[text_col].str.contains("hospital",case=False)|                            
    #                             dataset[text_col].str.contains(r"\bER\b",case=False)
    #                         )
    #                     )
    #                , cluster_col ] = "hospital"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("diff",case=False)|
                            dataset[text_col].apply(fuzzy_match, pattern="(?:distinguish){e<=3}")
                        )                
                    , cluster_col ] = "covid-versus"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("tested",case=False)|
                            dataset[text_col].str.contains("test",case=False)
                        )                
                    , cluster_col ] = "personal-testing"

    dataset.loc[
                        (dataset[cluster_col]=="personal-testing") & 
                        (
                            dataset[text_col].str.contains("(?:tested|test)",case=False) &
                            dataset[text_col].str.contains("where",case=False)
                        )                
                    , cluster_col ] = "personal-testing-location"

    dataset.loc[
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("recover",case=False)
                        )                
                    , cluster_col ] = "covid-recovery"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("at risk",case=False)|
                            (
                                dataset[text_col].str.contains("more",case=False) &
                                dataset[text_col].str.contains("risky|dangerous",case=False)
                            )
                        )                
                    , cluster_col ] = "covid-vulnerable"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            dataset[text_col].str.contains("dangerous",case=False)|
                            dataset[text_col].str.contains("risk",case=False)
                        )                
                    , cluster_col ] = "covid-contagious"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (dataset[text_col].str.contains(r"\bsymptom",case=False))                
                    , cluster_col ] = "covid-symptoms"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                dataset[text_col].apply(fuzzy_match, pattern="(?:whats|what (?:is|s))")
                            ) & 
                            (
                                dataset[text_col].apply(fuzzy_match, pattern="(?:covid){e<=2}")|
                                dataset[text_col].str.contains("corona",case=False)
                            )
                        )                
                    , cluster_col ] = "covid-what"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                (
                                    dataset[text_col].str.contains("how",case=False) &
                                    dataset[text_col].str.contains("long",case=False)
                                )|
                                dataset[text_col].str.contains("when",case=False)
                            )&
                                dataset[text_col].str.contains("will",case=False)&
                            (
                                dataset[text_col].str.contains("last|end|over|normal|done",case=False)
                            )
                        )                
                    , cluster_col ] = "situation-future"

    dataset.loc[                     
                        (dataset[cluster_col]=="unclassified") & 
                        (
                            (
                                dataset[text_col].str.contains("how|when|where",case=False) 
                            )&
                                dataset[text_col].str.contains("did",case=False)&
                            (
                                dataset[text_col].str.contains("start|begin|began",case=False)
                            )
                        )                
                    , cluster_col ] = "situation-past"

    return dataset