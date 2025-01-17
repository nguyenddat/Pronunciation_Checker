import abc
import numpy as np

class IASRModel(metaclass = abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "getTranscript") and
                callable(subclass.getTranscript) and
                hasattr(subclass, "getWordLocations") and
                callable(subclass.getWordLocations) and
                hasattr(subclass, "processAudio") and
                callable(subclass.processAudio))
    
    @abc.abstractmethod
    def getTranscript(self) -> str:
        raise NotImplementedError
    
    @abc.abstractmethod
    def getWordLocations(self) -> list:
        raise NotImplementedError
    
    @abc.abstractmethod
    def processAudio(self, audio):
        raise NotImplementedError

