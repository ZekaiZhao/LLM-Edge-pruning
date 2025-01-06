# LLM-Edge-pruning

Here, we have files for doing few shot inference on original unpruned Llama 3.2 1b instruct model on the GMS8K test dataset and file name is called original_llama1b_few_shot_inference.py

Doing edge pruning on the Llama 3.2 1b instruct model, we can use Llama_3.2_1b_edge_pruning.py. In order to perform the edge-pruning, we need to load functions from fllama_boolean_expressions_fs.py, I0_fllama.py, and modeling_fllama.py. The file Llama_3.2_1b_edge_pruning.py contains the code to load the functions and perform edge pruning on the interested model. 
