from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

tagger = SequenceTagger.load("dchaplinsky/flair-uk-ner")
splitter = SegtokSentenceSplitter()

text = """Зазвичай по повітряній цілі випускається 2 зенітні ракети, аби було більше шансів її знищити. Противник запускає по Україні новітні ракети – Х-101 випускалася рік-два тому… Знаходили уламки, де ракети 2019-20 року випуску, а по ній летить ракета 70-го року випуску (зенітна ракета С-300 чи Бук).

Ігнат наголосив при цьому, що все ж остаточне рішення ухвалюють командири на місцях.

Він нагадав, що 15 листопада РФ здійснила масивний удар по критичній інфраструктурі України.

Лише в західному регіоні було збито не менше 15 ракет противника, тобто ППО України теоретично могла використати 30 ракет.

Речник Повітряних сил наголосив, що це була потужна атака РФ та великий протиповітряний бій.
"""

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
Sentence: "Зазвичай по повітряній цілі випускається 2 зенітні ракети , аби було більше шансів її знищити ."
---
Sentence: "Противник запускає по Україні новітні ракети – Х-101 випускалася рік-два тому … Знаходили уламки , де ракети 2019-20 року випуску , а по ній летить ракета 70-го року випуску ( зенітна ракета С-300 чи Бук ) ." → ["Україні"/LOC, "Бук"/ORG]
---
entity.text is: "Україні"
entity.start_position is: "22"
entity.end_position is: "29"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9995115995407104"

entity.text is: "Бук"
entity.start_position is: "196"
entity.end_position is: "199"
entity "ner"-label value is: "ORG"
entity "ner"-label score is: "0.4689532220363617"

Sentence: "Ігнат наголосив при цьому , що все ж остаточне рішення ухвалюють командири на місцях ."
---
Sentence: "Він нагадав , що 15 листопада РФ здійснила масивний удар по критичній інфраструктурі України ." → ["РФ"/LOC, "України"/LOC]
---
entity.text is: "РФ"
entity.start_position is: "29"
entity.end_position is: "31"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9804568290710449"

entity.text is: "України"
entity.start_position is: "84"
entity.end_position is: "91"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9989952445030212"

Sentence: "Лише в західному регіоні було збито не менше 15 ракет противника , тобто ППО України теоретично могла використати 30 ракет ." → ["ППО України"/ORG]
---
entity.text is: "ППО України"
entity.start_position is: "72"
entity.end_position is: "83"
entity "ner"-label value is: "ORG"
entity "ner"-label score is: "0.9406674802303314"

Sentence: "Речник Повітряних сил наголосив , що це була потужна атака РФ та великий протиповітряний бій ." → ["Повітряних сил"/ORG, "РФ"/LOC]
---
entity.text is: "Повітряних сил"
entity.start_position is: "7"
entity.end_position is: "21"
entity "ner"-label value is: "ORG"
entity "ner"-label score is: "0.926283448934555"

entity.text is: "РФ"
entity.start_position is: "58"
entity.end_position is: "60"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9905871748924255"

Sentence: ""
---
"""
