from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("dchaplinsky/flair-uk-ner")

sentence = Sentence("Я люблю Україну. Моє імʼя Марія Шевченко, я навчаюся в Київській політехніці.")

tagger.predict(sentence)

print(sentence)
print('---')

print('The following NER tags are found:')
for entity in sentence.get_spans('ner'):
    print(f'entity.text is: "{entity.text}"')
    print(f'entity.start_position is: "{entity.start_position}"')
    print(f'entity.end_position is: "{entity.end_position}"')

    print(f'entity "ner"-label value is: "{entity.get_label("ner").value}"')
    print(f'entity "ner"-label score is: "{entity.get_label("ner").score}"\n')

# Result:
"""
Sentence: "Я люблю Україну . Моє імʼя Марія Шевченко , я навчаюся в Київській політехніці ." → ["Україну"/LOC, "Марія Шевченко"/PERS, "Київській політехніці"/ORG]
---
The following NER tags are found:
entity.text is: "Україну"
entity.start_position is: "8"
entity.end_position is: "15"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9993981122970581"

entity.text is: "Марія Шевченко"
entity.start_position is: "26"
entity.end_position is: "40"
entity "ner"-label value is: "PERS"
entity "ner"-label score is: "0.9854105412960052"

entity.text is: "Київській політехніці"
entity.start_position is: "55"
entity.end_position is: "76"
entity "ner"-label value is: "ORG"
entity "ner"-label score is: "0.9557965695858002"
"""
