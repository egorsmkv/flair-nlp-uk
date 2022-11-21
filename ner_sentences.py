from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

tagger = SequenceTagger.load("dchaplinsky/flair-uk-ner")
splitter = SegtokSentenceSplitter()

text = "Я люблю Україну. Моє імʼя Марія Шевченко, я навчаюся в Київській політехніці."

sentences = splitter.split(text)

tagger.predict(sentences)

print('The following NER tags are found:')
for sentence in sentences:
    print(sentence)
    print('---')

    for entity in sentence.get_spans('ner'):
        print(f'entity.text is: "{entity.text}"')
        print(f'entity.start_position is: "{entity.start_position}"')
        print(f'entity.end_position is: "{entity.end_position}"')

        print(f'entity "ner"-label value is: "{entity.get_label("ner").value}"')
        print(f'entity "ner"-label score is: "{entity.get_label("ner").score}"\n')

# Result:
"""
The following NER tags are found:
Sentence: "Я люблю Україну ." → ["Україну"/LOC]
---
entity.text is: "Україну"
entity.start_position is: "8"
entity.end_position is: "15"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9994187355041504"

Sentence: "Моє імʼя Марія Шевченко , я навчаюся в Київській політехніці ." → ["Марія Шевченко"/PERS, "Київській політехніці"/ORG]
---
entity.text is: "Марія Шевченко"
entity.start_position is: "9"
entity.end_position is: "23"
entity "ner"-label value is: "PERS"
entity "ner"-label score is: "0.9852352738380432"

entity.text is: "Київській політехніці"
entity.start_position is: "38"
entity.end_position is: "59"
entity "ner"-label value is: "ORG"
entity "ner"-label score is: "0.957990288734436"
"""
