
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
