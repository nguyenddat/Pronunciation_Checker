import torch

from ..schemas import ITranslationModel

class NeuralTranslator(ITranslationModel.ITranslationModel):
    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def translateSentence(self, sentence: str) -> str:
        tokenized_text = self.tokenizer(sentence, return_tensors = "pt")
        translation = self.model.generate(**tokenized_text)
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens = True
        )[0]

        return translated_text