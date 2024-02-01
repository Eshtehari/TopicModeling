from typing import Union

from bertopic import BERTopic
from gensim.models import LdaModel


class TopicDiversity():
    """
    Original authors: https://github.com/MIND-Lab/OCTIS/
    Editor: Shayan
    """
    
    def __init__(self, topk=10):
        """
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        """
        self.topk = topk


    def score(self, model_output: Union[LdaModel, BERTopic], model='gensim'):
        """
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        """
        if model == 'gensim':
            # model_output: LdaModel # doesn't work in if statement!
            # topics = model_output["topics"] # Error, doesn't work anymore?
            topics_terms = [model_output.get_topic_terms(topic) for topic in range(0, model_output.num_topics)]
            # topics = [topic.sort(key=lambda k: k[1], reverse=True) for topic in topics_terms] # they are already ordered
            topics = [[idx for idx, prob in topic] for topic in topics_terms]
        else:
            # model_output: BERTopic
            topics_terms = list(model_output.get_topics().values())
            topics = [[term for term, prob in topic] for topic in topics_terms]
        
        
        if (topics is None) or (len(topics) == 0):
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td
