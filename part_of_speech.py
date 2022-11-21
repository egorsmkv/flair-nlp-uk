from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("dchaplinsky/flair-uk-pos")

sentence = Sentence("Я люблю Україну. Моє імʼя Марія Шевченко, я навчаюся в Київській політехніці.")

tagger.predict(sentence)

print(sentence)
print('---')

print('The following POS tags are found:')
for entity in sentence.get_spans('pos'):
    print(entity)

# Result:
"""

"""
