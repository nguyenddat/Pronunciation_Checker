import abc

class ITranslationModel(metaclass = abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "translateSentence") and
                callable(subclass.translateSentence))
    
    @abc.abstractmethod
    def translateSentence(self, str) -> str:
        raise NotImplementedError

