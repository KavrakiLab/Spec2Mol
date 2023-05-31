# Running the Decoder


The decoder can be run as 

```bash
python -m scripts.decode_embeddings --output_file decoded_output.csv \
				    --predicted_embeddings sample.pt \
				    --model translation \
				    --device cuda \
				    --model_load models/model.pt \
				    --vocab_load models/vocab.nb \
				    --config_load models/config.nb \
				    --n_batch 65 \
				    --num_variants 3 
```

`output_file`: A csv file where the output will be saved. Multiple SMILES strings will be generated for each embedding and will be separated using the `|` symbol.

`predicted_embeddings`: The embeddings predicted by the spectra encoder. The decoder expects a python dictionary with the SMILES representation of the molecule as the key, and a tensor of the predicted embeddings as the value. If the SMILES representation of the spectra is unknown, any unique string can be used as a key. The dictionary should be saved to a file using torch.save method. A sample embeddings file is provided (sample.pt)

`model` : The specific model identifier. Should always be 'translation'

`device`: cuda or cpu. cuda is recommended

`model_load`: Path to the model

`vocab_load`: Path to the vocabulary

`config_load`: Path to the model config

`n_batch`: Batch size

`num_variants`: The number of SMILES strings to generate for each embeddings.

We would like to thank the Moses project https://github.com/molecularsets/moses. The translation model code has been modified from this project.
