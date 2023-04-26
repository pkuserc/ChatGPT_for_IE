​		You should convert the original ACE2005 format data to the data format in `ACE2005_ChatGPT_answers_SAMPLE.json` file, which is

```json
{
    "sentences": [["first", "an", "update", "on", "a", "long", "running", "air", "safety", "investigation", "a", "year", "and", "a", "half", "near", "an", "airline", "crashed", "near", "new", "york", "'s", "kennedy", "airport", "there", "is", "controversy", "whether", "the", "disaster", "could", "have", "been", "averted", "."]],
    "ner": [[[17, 17, "VEH"], [23, 24, "FAC"], [20, 21, "GPE"]]],
    "relations": [[[23, 24, 20, 21, "PART-WHOLE"]]],
    "clusters": [],
    "doc_key": "",
    "predicted_ner": [[[20, 21, "GPE"], [23, 24, "FAC"]]],
    "predicted_relations": [[[20, 21, 23, 24, "PART-WHOLE"]]]
}
```

where `predicted_ner` and `predicted_relations` are from answers by ChatGPT. The prompts are

```json
{
    "prompts": {
        "predicted_ner": "Sentence: '{SENT}'\nEntity label set: '{NERID2LABLE}'\nBased on the given label set, please extract the entities and their types from the given text. Answer me in format by [{\"Entity Name\": \"Entity Label\"}] without any additional things including your explanations or notes.",
        "predicted_relations": "Sentence: '{SENT}'\nRelation label set: '{RELID2LABEL}'\nGiven Entities in the Sentence: '{PREDNER}'\nBased on the given label set, please extract the related entity pairs from the given entities. Answer me in format by [[\"Head Entity\", \"Tail Entity\", \"Relation Type\"]] without any additional things including your explanations or notes."
    }
}
```

where,

```python
NERID2LABLE = ["GPE", "PERSON", "VEHICLE", "ORG", "FACILITY", "LOC", "WEAPON"]
RELID2LABEL = ["artifact", "Gen-affiliation", "METONYMY", "Org-affiliation", "part-whole", "person-social", "physical"]
```

Then, we can transform the raw data format into

```json
{
    ...
    "predicted_ner": [[[20, 21, "GPE"], [23, 24, "FAC"]]],
    "predicted_relations": [[[20, 21, 23, 24, "PART-WHOLE"]]]
}
```
