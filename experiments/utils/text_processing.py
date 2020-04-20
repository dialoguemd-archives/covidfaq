# Dataperformer 2019 - Stellar Server AI - Version 1.0
# File		  	: text_processing.py
# Author		: Sunanda
# Link		 	: 
# Creation date : 29/07/2019
# Last update   : 22/08/2019
# Description   : For common methods for text processing
# Dependencies	: NLTK, NLTK punkt tokenizer, NLTK Stopwords

'''
@ Todo - 
	- Add Keras methods, Pandas Dataframe Handlers
	- Add multiprocesing where you can. (The whole thing needs to be re-arranged for that)
'''

import csv
import json
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
import numpy as np 
import keras
import pandas as pd

class Preprocessing:	
	'''
	Preprocessing Summary
	This class is expected to provide necessary preprocessing functions and attributes

	Coming up Next
		- 	Set a frequency limit. Any tokens with a frequency less than the specified number will not
			be add to the tokenized output
		- 	if removeMultipleOccurrences=True, then we are not getting correct token frequency counts
		- 	Word index, word count, document count attributes similar to the one provided by Keras
		
	Attributes:
		dataset_obj 				: Object of the Dataset Class
		removeMultipleOccurrences 	: Removes multiple occurrence os the tokens within a document if 
									  set to True
		removeStopwords 			: Removes Stopwords from the output if set to True, takes the default 
									  nltk english stopwords list into account if a stopwords list is not
									  provided
		stopwordsList				: Stopword List to be used for stopword removal
		minFrequency 				: Set a minimum frequency requirement in order to be indexed
		language 					: The language for default nltk stowords list to be used. 
									  Default = "english" 
	'''

	# term : {DocID : number of times it appeared in the document}
	inverted_index = {}

	def __init__(self,
				 dataset_obj = None,
				 removeMultipleOccurrences = False, 
				 removeStopwords = False, 
				 stopwordsList = None,
				 minFrequency = None,
				 frequencyLimitFilter = "document", # "total"
				 track = None,
				 language="english"):
		self.dataset_obj = dataset_obj
		self.removeMultipleOccurrences = removeMultipleOccurrences
		self.removeStopwords = removeStopwords
		self.language = language
		self.track = track
		# Remove words based on their document frequency or total frequency
		self.frequencyLimitFilter = frequencyLimitFilter
		self.minFrequency = minFrequency
		self.word_index = {}
		self.word_count = {}
		self.doc_count = {}

		# if stopwords are to be removed but no specific list is provided 
		if removeStopwords and stopwordsList is None:		
			self.stopwordsList = nltk.corpus.stopwords.words(self.language)

	# Under Development - Rank score value
	def rsv(docID):
	    rsv = 0
	    for term in results:
	        if docID in results[term]:            
	            df = len(results[term])
	            informativeness = math.log(N/df)
	            tf=results[term][docID]
	            relevance_numerator = (k+1)*tf
	            relevance_denominator = k*((1-b)+b*(Ld[str(docID)]/Lavg))+tf
	            relevance = relevance_numerator / relevance_denominator            
	            rsv += (informativeness * relevance)
	        else:
	            rsv += 0
	    return rsv

	def get_word_index(self):
		if self.track : self.track.update("Generating Word Index")
		# term : rank (if sorted in descedning order by frequency)
		self.word_index={}
		# @To-do The Frequency updates nead to be in the Text_Tokenizer

		word_count_list = [[word, count] for word, count in self.get_word_count().items()]
		word_count_list = sorted(word_count_list,key=lambda x: x[1],reverse=True)
		for i, (word, _) in enumerate(word_count_list,1):
			self.word_index[word] = i
		return self.word_index

	# @property
	# def word_count(self):
	# 	return self.get_word_count()

	# @property	
	# def doc_count(self):
	# 	return self.get_doc_count()

	def get_word_count(self,tokenized_texts=None):		
		# if self.track : self.track.update("Calculating Word (Total) Frequencies")
		self.word_count={}
		if not tokenized_texts:
			tokenized_texts = self.dataset_obj.tokenized_texts
		for DocID, tokenized_text in zip(self.dataset_obj.ids,tokenized_texts):
			for word in tokenized_text:
				if word in self.word_count:							
					self.word_count[word]+= 1
				else:
					self.word_count[word] = 1
		return self.word_count

	def get_doc_count(self,tokenized_texts=None):
		# if self.track : self.track.update("Calculating Word (Document) Frequencies")
		self.doc_count={}		
		if not tokenized_texts:
			tokenized_texts = [list(set(tokenized_text)) for tokenized_text in self.dataset_obj.tokenized_texts]
		for DocID, tokenized_text in zip(self.dataset_obj.ids,tokenized_texts):
			for word in tokenized_text:
				if word in self.doc_count:		
					self.doc_count[word] += 1
				else:		
					self.doc_count[word] = 1
		return self.doc_count

	@property	
	def inverted_index(self):
		# if self.track : self.track.update("Generating Inverted Index")
		self._inverted_index = {}
		# The Frequency updates nead to be in the Text_Tokenizer
		for DocID, document in self.dataset_obj.dataset.items():
			for word in document["tokenized"]:
				if word in self._inverted_index:
					if DocID in self._inverted_index[word]:								
						self._inverted_index[word][DocID] += 1
					else:
						self._inverted_index[word][DocID] = 1
				else:
					self._inverted_index[word] = {DocID:1}
		return self._inverted_index


	def Text_Tokenizer(self,text):
		try:
			tokenized_text = nltk.tokenize.word_tokenize(text.lower())
		except:
			raise Exception("Error for text \n {}".format(text))
		processed_text = []
		for token in tokenized_text:
			if token.isalnum():
				if not self.removeStopwords or token not in self.stopwordsList:
					processed_text.append(token)

		if self.removeMultipleOccurrences:
			processed_text = list(set(processed_text))
		return processed_text

	def Bulk_Tokenizer(self, list_of_texts):
		'''
		Bulk_Tokenizer Summary
		To be used in the case of a bundle (list) of texts
		'''
		pool = mp.Pool(mp.cpu_count())		
		list_of_tokenized_texts = pool.map(self.Text_Tokenizer, [text for text in list_of_texts])	
		pool.close()	
		return list_of_tokenized_texts

	def Tokenize(
					self, 
					texts=[], 
					add_to_col="tokenized", 
					returnFlag=False, 
					useKeras=False, 
					num_words=None, 
					removeStopwords=None, 
					track=None
				):
		'''
		Tokenize Summary
		To be used with the object of dataset class
		'''		

		pool = mp.Pool(mp.cpu_count())

		if not texts:
			texts=self.dataset_obj.texts

		if removeStopwords != None: 
			self.removeStopwords = removeStopwords

		if self.removeStopwords and self.stopwordsList is None:		
			self.stopwordsList = nltk.corpus.stopwords.words(self.language)

		if track != None:
			self.track = track

		if not self.dataset_obj:
			raise ValueError('Dataset class object expected but not initialized')

		if useKeras:
			if self.track : self.track.update("Tokenizing with Keras")
			self.keras_tokenizer_obj = keras.preprocessing.text.Tokenizer(num_words=num_words)
			self.keras_tokenizer_obj.fit_on_texts(texts)
			self.word_index = self.keras_tokenizer_obj.word_index
			self.word_count = self.keras_tokenizer_obj.word_counts
			self.doc_count = self.keras_tokenizer_obj.word_docs
			self.rank_index = {rank: word for word, rank in self.word_index.items()}

			if self.track : self.track.update("Texts to sequences with Keras")
			list_of_texts_sequences = self.keras_tokenizer_obj.texts_to_sequences(texts)
			
			# Lambda to mutiprocess sequence to tokens			
			if self.track : self.track.update("Sequences to Tokens")
			list_of_tokenized_texts =  pool.map(self.sequence_to_tokens,list_of_texts_sequences)
			
			if self.removeStopwords:	
				if self.track : self.track.update("Removing Stopwords")
				# Remove Stopwords
				cleaned_tokenized_texts =  pool.map(self.remove_stopwords,list_of_tokenized_texts)
				list_of_tokenized_texts = cleaned_tokenized_texts

		else:			
			if self.track : self.track.update("Tokenizing")
			list_of_tokenized_texts = pool.map(self.Text_Tokenizer, [text for text in texts])	
			
		if self.frequencyLimitFilter == "document":
			self.doc_count = self.get_doc_count(list_of_tokenized_texts)
		elif self.frequencyLimitFilter == "total":
			self.word_count = self.get_word_count(list_of_tokenized_texts)

		if self.minFrequency:
			if self.frequencyLimitFilter == "document":
				self.freq_dict = self.doc_count
			elif self.frequencyLimitFilter == "total":
				self.freq_dict = self.word_count
			else:
				raise ValueError('frequencyLimitFilter argument value can only be "document" or "total"')

			if self.track : self.track.update("Removing Low Frequency Words")
			if self.track : self.track.print("Vocabulary size (BEFORE) - ",len(self.freq_dict))
			cleaned_tokenized_texts = pool.map(self.remove_low_frequency_words, list_of_tokenized_texts)
			list_of_tokenized_texts = cleaned_tokenized_texts
			if self.track : self.track.print("Vocabulary size (After) - ",len(self.get_word_count(list_of_tokenized_texts)))
		
		if self.track : self.track.update("Saving Tokenized Texts to Dataset object for later use")
		
		# Saving to Dataset	object (Overwrite previous one)
		self.dataset_obj.add_to_dataset(list_of_tokenized_texts, add_to_col)
		pool.close()	

		if returnFlag:
			return list_of_tokenized_texts

		return self


	def keras_texts_to_matrix(self, texts=[], mode="binary", label="binary_vectors"):
		# @Todo - Remove Stopwords
		if not texts:
			texts=self.dataset_obj.texts

		vectors = self.keras_tokenizer_obj.texts_to_matrix(texts, mode=mode)	
		remove_these_columns = []
		remove_these_words = []
		if self.removeStopwords:
			for word, rank in self.word_index.items():
				if word in self.stopwordsList:
					remove_these_columns.append(rank-1)
					remove_these_words.append(word)
					# del self.word_index[word]
		vectors = np.delete(vectors,remove_these_columns, axis=1)
		# New word index
		w_i = {word: rank for word, rank in self.word_index.items() if word not in remove_these_words}
		w_i_sorted = sorted(w_i.items() ,  key=lambda x: x[1])
		for i, (word, _) in enumerate(w_i_sorted,1):
			w_i.update({word:i})
		self.word_index = w_i
		self.dataset_obj.add_to_dataset(vectors,label)
		return self

	# Inspired from Keras Tokenizer
	def texts_to_sequences(self):
		return
	def sequences_to_matrix(self, mode="binary"):
		return
	def texts_to_matrix(self, mode="binary"):
		# inverted_index = self.inverted_index
		if self.track : self.track.update("Generating Matrix")
		word_index = self.get_word_index()
		vectors = np.zeros((len(self.dataset_obj), len(word_index)))
		for row_num, tokens in enumerate(self.dataset_obj.tokenized_texts):
			for token in tokens:	
				vectors[row_num][word_index[token]-1]=1
		self.dataset_obj.add_to_dataset(vectors,"binary_vectors")
		return self

	def remove_stopwords_keras(self):
		return self

	def NER(self):
		'''
		Coming up next 
		Named Entities Recognition 
		'''
		return self

	def save_word_weights(self, save_as = None):
		word_counts = [[word,count] for word,count in self.word_freq.items()]
		# Sort by count
		word_counts = sorted(word_counts,key=lambda x: x[1],reverse=True)	 
		word_counts = [" ".join([word,str(count)]) for word, count in word_counts]
		word_counts = "\n".join(word_counts)
		# Saving Word Index for Reference
		with open(save_as, 'w+') as word_weights_file:
			word_weights_file.write(word_counts)

	def sequence_to_tokens(self, sequence):
		return [self.rank_index[num] for num in sequence]

	def remove_stopwords(self, tokens):
		return [token for token in tokens if token not in self.stopwordsList]

	def remove_low_frequency_words(self,tokens):
		return [token for token in tokens if self.freq_dict[token]>self.minFrequency]

class Dataset:
	"""
	Dataset Summary
	This class is expected to provide with dataset related functionalities 
	assuming it is provided with the location of CSV file with headers.
 
	Attributes:  	
		has_header 		If not specified, it is assumed that the dataset (csv file)
						has a header
		ids 			A list of all the socument IDs sorted in ascending order
		texts 			A list of all the body/text/content of the article sorted in
						the order of the document ID
		dataset 		It is the dataset stored in the form of a dictionary
						dataset[docID][header] = value
						For example, dataset[3611]["title"] = "A new survey"
						Implies that for document with document ID 3611, the title 
						column value is "A new survey" 
	 	file_location	The location/path of the dataset file
	 	title_header	The header name of the title of the article
	 	text_header 	The header name of the text/body/content of the article

	Methods:
		load()
	"""   
	

	
	def __init__(self, 
				 file_location, 
				 has_header = True, 
				 manual_headers=None,
				 auto_indexing=False,
				 index_label="id", 
				 title_header="title",
				 text_header="text",
				 max_num = None):
		self.file_location = file_location
		self.has_header = has_header
		self.manual_headers = manual_headers
		self.title_header = title_header
		self.text_header = text_header
		self.max_num = max_num
		self.processing_obj = Preprocessing(self)
		self.auto_indexing = auto_indexing
		self.index_label = index_label

		# Flags
		self.has_header = True

		# Initializing Variables
		self.ids = []
		# self.texts = []
		self.dataset = {}

	def __getitem__(self,val):
		if isinstance(val, slice):
			start, stop, step = val.indices(len(self))
			if start < 0 : start = len(self) + start
			sliced = []
			for i, DocID in enumerate(self.ids):
				if stop is not None and stop == i:
					# Expected to stop somewhere and here we are
					break
				elif start is None or start <= i:
					# Expected to start now
					sliced.append([val for _,val in self.dataset[DocID].items()])
			return sliced
		elif isinstance(val, int):
			return [val for val in self.dataset[self.ids[val]]]
		elif isinstance(val, str):
			# Assuming Doc ID
			return [val for val in self.dataset[val]]

	# def __print__():
	# 	return

	def load(self):
		"""
		This function currently works specifically for the dataset
		that we have. So it makes certain assumptions like the language
		is specified in the 9th Column of the row, etc.
		"""
		with open(self.file_location) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for i, row in enumerate(csv_reader):
				# If first row is a header
				if i == 0:
					if self.has_header:
						self.headers = row
					else:
						self.headers=list(range(len(row)))
					
				# Only for documents in specified language
				else:
					if self.auto_indexing:
						docID = i
					else:
						if self.index_label in self.headers:
							# From string to int
							docID = int(row[self.headers.index(self.index_label)])
						else:
							raise Exception("Index Label not found")

					# Saving Doc IDs seperately
					self.ids.append(docID)

					# Saving the whole row as a dictionary
					self.dataset[docID] = {}
					for header,val in zip(self.headers,row):
						self.dataset[docID][header] = val

				# Process them no more		
				if self.max_num == i:
					break

		# sort ids list in ascending order of ids
		self.ids.sort()
		return self

	def get(self,header):
		self._header_contents = []
		for docID in self.ids:
			self._header_contents.append(self.dataset[docID][header])
		return self._header_contents

	@property
	def texts(self):
		self._texts = []
		for docID in self.ids:
			self._texts.append(self.dataset[docID][self.text_header])
		return self._texts

	@property
	def tokenized_texts(self):
		self._tokenized_texts = []
		for docID in self.ids:
			self._tokenized_texts.append(self.dataset[docID]["tokenized"])
		return self._tokenized_texts

	def __len__(self):
		return len(self.ids)

	@property
	def word_count(self):
		return self.processing_obj.get_word_count()
	

	def filter(self, filter_header="language", filter_val = "en"): 
		'''
		By default filters the dataset and keeps only english articles
		but can be used to filter for any value under any header
		'''
		filtered_dataset = {}
		for docID, document in self.dataset.items():
			if document[filter_header] == filter_val:
				filter_header[docID] = document
		self.dataset = filtered_dataset
		return self

	def Tokenize(self,col=None,**args):
		if not col:
			col=self.text_header
		texts_to_tokenize=self.get(col)
		if "returnFlag" in args:
			if args["returnFlag"]:
				tokenized_text = self.processing_obj.Tokenize(texts=texts_to_tokenize,**args)
				return tokenized_text
		else:
			self.processing_obj.Tokenize(**args)
		return self

	def save_word_weights(self, save_as):
		word_counts = [[word,count] for word,count in self.word_count.items()]
		# Sort by count
		word_counts = sorted(word_counts,key=lambda x: x[1],reverse=True)	 
		word_counts = [" ".join([word,str(count)]) for word, count in word_counts]
		word_counts = "\n".join(word_counts)
		# Saving Word Index for Reference
		with open(save_as, 'w+') as word_weights_file:
			word_weights_file.write(word_counts)

	def add_to_dataset(self, values, label):
		# Assumed to be sorted by DocID by default
		for docID, value in zip(self.ids,values):
			self.dataset[docID][label] = value
		return self


	def set_vectors(self, vectors, label = "vector"):
		self.vector_label = label
		# Assumed to be sorted by DocID by default
		for docID, doc_vec in zip(self.ids,vectors):
			self.dataset[docID][self.vector_label] = doc_vec
		return self

	# def dataset_analysis(self):
	# 	for word, count in self.word_freq

	def plot_zirfs_law(self):
		x = []
		y = []
		for word, rank in self.word_index.items():
			x.append(rank)
			y.append(math.log(int(self.word_freq[word])))
		plt.plot(x,y,"r.")
		plt.xlabel('Word Rank')
		plt.ylabel('Log of Word Frequency')
		plt.title('Plot of word frequency')
		# plt.grid(True)
		# display the plotscs
		plt.show()

"""
	def remove_duplicates(self, save_as = None):
		return

	def split_into_paras(self, save_as = None):
		return

	def save(self):
		return

	def save_as(self, location):
		return

	def save_paras(self, location):
		return

	def analyse(self):
		self.null_val_count = {}
		for docID, data_dict in dataset.items():
			for header, value in data_dict.items():
				if value == "NULL":
					if header in null_val_count: null_val_count[header] += 1
					else: null_val_count[header] = 1
		self.null_value_dict = null_val_count
		return

	def plot_zirfs_law(self):
		return

	def analyse_null_values(self):
		return

"""
