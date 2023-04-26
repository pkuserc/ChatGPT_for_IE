
## Dataset

#### `ACE05-E`

1. Download ACE2005 dataset from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)
2. Put data processed by [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event) into the folder `data/ACE05-E`
3. Run `python ./Code/convert2sentence.py`

#### `ACE05-E+`

1. Download ACE2005 dataset from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)
2. Put data processed by [OneIE](http://blender.cs.illinois.edu/software/oneie/) into the folder `data/ACE05-E+`

## Generate prompts for event datasets
run `sh gen_event_prompt.sh.`, generated prompts will be saved in the `Prompt` dir.

## Get output from ChatGPT

run `python ./Code/call_api.py --task_dataset=ED_E_Closed`, where `task_dataset` argument can be "EAE_E_Open", "EE_E+_Closed", etc ...

All outputs returned from ChatGPT are in `Output` dir. File names in `Output` dir follows this format: `{doc_id}-{sent_id}` or `{doc_id}-{sent_id}-{event_id}`

## Evaluation
run `python ./Code/ED/score_ED_E.py` and `./Code/ED/score_ED_E+.py` to evaluate ED task on ACE05-E and ACE05-E+, respectively. The same for EAE and EE tasks.
