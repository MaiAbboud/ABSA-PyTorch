import nltk
import pandas as pd
import spacy
import re


# from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# from tqdm.notebook import tqdm
# import csv
from split_dataset import split_dataset

nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Preprocssing coursera dataset 
class Preprocessing:
    def __init__(self,dataset,
                 negation_handling_on=True,
                 lowercase_on=True,
                 stop_word_delete="all",
                 punctuation_on=True,
                 repeated_on=True,
                 mention_on=True,
                 merge_num_on=True,
                 remove_single_on=True,
                 merge_on=True,
                 stem_on=True,
                 lemma_on=True):
        
        """ dataset (str): The input text dataset.
            negation_handling_on (bool): whether to add prefix to words after negation.
            lowercase_on (bool): Whether to convert to lowercase.
            stop_words_on (bool): Whether to remove stop words.
            punctuation_on (bool): Whether to remove punctuation.
            repeated_on (bool): Whether to handle repeated letters.
            mention_on (bool): Whether to remove mentions and URLs.
            merge_num_on (bool): Whether to merge numerical values.
            remove_single_on (bool): Whether to remove single-letter words.
            stem_on (bool): Whether to apply stemming.
            lemma_on (bool): Whether to apply lemmatization."""
        
        self.dataset = dataset
        self.negation_handling_on=negation_handling_on
        self.lowercase_on=lowercase_on
        self.stop_word_delete=stop_word_delete
        self.punctuation_on=punctuation_on
        self.repeated_on=repeated_on
        self.mention_on=mention_on
        self.merge_num_on=merge_num_on
        self.remove_single_on=remove_single_on
        self.merge_on=merge_on
        self.stem_on=stem_on
        self.lemma_on=lemma_on

        self.negation_words = {"not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", 
        "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", 
        "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", 
        "hadn't", "mustn't", "couldn't", "mightn't", "shan't", "cannot", "without"}
    
    def negation_handling(self):
        tokens = self.dataset.split()
        negated = False
        result = []

        for token in tokens:
            # Check if token is a negation word
            if token in self.negation_words:
                negated = True
                result.append(token)
            elif negated:
                # Apply a prefix to mark the word as negated
                result.append("neg " + token)
                # Stop negation at punctuation or sentence end
                if re.search(r'[,.!?]|but|however', token):
                    negated = False
            else:
                result.append(token)
        self.dataset = " ".join(result)
        
    def lower_case_letters(self):
        """Convert a dataset to lowercase."""
        dataset = word_tokenize(self.dataset.lower())
        self.dataset = ' '.join(dataset)

    def delete_stop_words(self,stop_word_delete):
        # if stop_words_on:
        stop_words = {}
        if stop_word_delete == "all":
            stop_words = set(stopwords.words('english'))
        if stop_word_delete == "all_except_neg":
            stop_words = set(stopwords.words('english'))
            # Exclude negation words from stop words
            stop_words = stop_words - self.negation_words
        if stop_word_delete =="none":
            stop_words = {}

        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(self.dataset)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words ]
        self.dataset = ' '.join(filtered_tokens)
    
    def delete_punctuation(self):
        """Remove punctuation from a dataset."""
        self.dataset = re.sub(r'[,:;!-\./(){}[]|؟]', ' ', self.dataset)

    def delete_repeated(self):
        """Remove repeated letters in a dataset."""
        self.dataset = re.sub(r'(.)\1{2,}', r'\1', self.dataset)
        """Remove repeated words in a dataset."""
        self.dataset = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', self.dataset)
        
    def delete_mention(self):
        """Remove mentions and URLs from a dataset."""
        # if mention_on:
        self.dataset = re.sub(r"@\S+", '', self.dataset)
        self.dataset = re.sub(r'https?\S+', '', self.dataset)

    def merge_num(self):
        """Merge numerical values in a dataset."""
        self.dataset = re.sub(r"[0-9]+(\.)?[0-9]+", "1", self.dataset)

    def remove_single_letter_words(self):
        """Remove single-letter words from a dataset."""
        pattern = r'\b[a-hj-z]\b'
        self.dataset = re.sub(pattern, '', self.dataset)

    def merge(self, aspects, aspect):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, aspects)) + r')\b', re.IGNORECASE)
        merged_dataset = pattern.sub(aspect, self.dataset)
        self.dataset = merged_dataset

    def stem(self):
        porter = PorterStemmer()
        dataset = word_tokenize(self.dataset)
        dataset = [porter.stem(word) for word in dataset]
        self.dataset = ' '.join(dataset)

    def lemma(self):
        lemmatizer = WordNetLemmatizer()
        dataset = word_tokenize(self.dataset)
        dataset = [lemmatizer.lemmatize(word) for word in dataset]
        self.dataset =  ' '.join(dataset)

    def preprocess(self):

        """Perform a series of preprocessing steps on a dataset.
        Returns:
            str: The preprocessed dataset.

        """
        aspect_1 = ["teacher", "teachers", "instructor", "instructors", "professors", "professor", "lecturer", "profs",
                    "prof", 'educator', 'tutor', 'tutors', 'mentor', 'mentors', 'coach', 'coaches', 'guide', 'guides',
                    'facilitator', 'facilitators', 'pedagogue', 'pedagogues', 'schoolmaster', 'schoolmasters',
                    'schoolmistress', 'preceptor', 'preceptors', 'academic', 'academics', 'trainer', 'trainers',
                    'counsellor', 'counsellors', 'adviser', 'advisers', 'pedagogist', 'pedagogists', 'didactic', 'master']

        aspect_2 = ["class", "courses", "course", "materials", "material", "syllabus", "homeworks", "exercise", "exercises",
                    'program', 'module', 'lecture', 'seminar', 'workshop', 'curriculum', 'subject', 'training', 'session',
                    'Sections', 'Section', 'lesson', 'unit', 'elective', 'tutorial', 'education', 'Path', 'track', 'programs',
                    'modules', 'lectures', 'seminars', 'workshops', 'Curricula', 'subjects', 'trainings', 'sessions',
                    'lessons', 'units', 'electives', 'tutorials', 'workshops', 'educations', 'Paths', 'tracks']
        
        if self.negation_handling_on:
            self.negation_handling()
        if self.lowercase_on:
            self.lower_case_letters()
        self.delete_stop_words(self.stop_word_delete)
        if self.punctuation_on:
            self.delete_punctuation()
        if self.repeated_on:
            self.delete_repeated()
        if self.mention_on:
            self.delete_mention()
        if self.merge_num_on:
            self.merge_num()
        if self.remove_single_on:
            self.remove_single_letter_words()
        if self.merge_on:
            self.merge(aspect_1, 'teacher')
            self.merge(aspect_2, 'course')
        if self.lemma_on:
            self.lemma()

        return self.dataset 
    
def preprocess_dataset(dataset, negation_handling_on,stop_word_delete):
    processor = Preprocessing(dataset, negation_handling_on=negation_handling_on,
                              stop_word_delete = stop_word_delete)
    return processor.preprocess()