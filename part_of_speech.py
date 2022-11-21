from flair.data import Sentence
from flair.models import SequenceTagger

from pprint import pprint

tagger = SequenceTagger.load("dchaplinsky/flair-uk-pos")

sentence = Sentence("Я люблю Україну. Моє імʼя Марія Шевченко, я навчаюся в Київській політехніці.")

tagger.predict(sentence)

print(sentence)
print('---')

print('The following POS tags are found:')

pos_items = []
for label in sentence.get_labels():
    all_labels = []
    keys = label.data_point.annotation_layers.keys()
    
    for key in keys:
        all_labels.extend(
            [
                {'label': label.value, 'score': round(label.score, 4)}
                for label in label.data_point.get_labels(key)
                if label.data_point == label
            ]
        )

    pos_items.append({
        'text': label.data_point.text,
        'all_labels': all_labels,
    })

pprint(pos_items)

# Result:
"""
Sentence: "Я люблю Україну . Моє імʼя Марія Шевченко , я навчаюся в Київській політехніці ." → ["Я"/PRON, "люблю"/VERB, "Україну"/PROPN, "."/PUNCT, "Моє"/DET, "імʼя"/NOUN, "Марія"/PROPN, "Шевченко"/PROPN, ","/PUNCT, "я"/PRON, "навчаюся"/VERB, "в"/ADP, "Київській"/ADJ, "політехніці"/NOUN, "."/PUNCT]
---
The following POS tags are found:
[{'all_labels': [{'label': 'PRON', 'score': 1.0}], 'text': 'Я'},
 {'all_labels': [{'label': 'VERB', 'score': 1.0}], 'text': 'люблю'},
 {'all_labels': [{'label': 'PROPN', 'score': 1.0}], 'text': 'Україну'},
 {'all_labels': [{'label': 'PUNCT', 'score': 1.0}], 'text': '.'},
 {'all_labels': [{'label': 'DET', 'score': 0.9999}], 'text': 'Моє'},
 {'all_labels': [{'label': 'NOUN', 'score': 1.0}], 'text': 'імʼя'},
 {'all_labels': [{'label': 'PROPN', 'score': 1.0}], 'text': 'Марія'},
 {'all_labels': [{'label': 'PROPN', 'score': 1.0}], 'text': 'Шевченко'},
 {'all_labels': [{'label': 'PUNCT', 'score': 1.0}], 'text': ','},
 {'all_labels': [{'label': 'PRON', 'score': 1.0}], 'text': 'я'},
 {'all_labels': [{'label': 'VERB', 'score': 1.0}], 'text': 'навчаюся'},
 {'all_labels': [{'label': 'ADP', 'score': 1.0}], 'text': 'в'},
 {'all_labels': [{'label': 'ADJ', 'score': 1.0}], 'text': 'Київській'},
 {'all_labels': [{'label': 'NOUN', 'score': 1.0}], 'text': 'політехніці'},
 {'all_labels': [{'label': 'PUNCT', 'score': 1.0}], 'text': '.'}]
"""
