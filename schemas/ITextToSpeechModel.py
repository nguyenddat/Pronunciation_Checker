import abc
import numpy as np

class ITextToSpeechModel(metaclass = abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "getAudioFromSentence") and
                callable(subclass.getAudioFromSentence))
    
    @abc.abstractmethod
    def getAudioFromSentence(self, str) -> np.ndarray:
        raise NotImplementedError

