import argparse
import warnings
import torch

import pandas as pd
from moses.models_storage import ModelsStorage
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(ListDataset).__init__()
        self.data = list(zip(*data))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def get_embeddings_tensor(embeddings_map, smiles_list):
    retval = []
    for smi in smiles_list:
        retval.append(embeddings_map[smi])
    retval = torch.stack(retval)
    #    print("retval.size()", retval.size())
    return retval


def decode(
    model, train_loader, predicted_embeddings, output_file, num_variants
):
    smiles_with_precursor = []

    predicted_smiles_variants = []
    for i in range(num_variants):
        predicted_smiles_variants.append([])

    for i, input_batch in enumerate(tqdm(train_loader)):
        sk = input_batch[0]
        mu_predicted = get_embeddings_tensor(predicted_embeddings, sk)
        mu_predicted = mu_predicted.cuda()
        smiles_with_precursor.extend(sk)

        for j in tqdm(range(num_variants)):
            predicted_smiles_batch_j = model.sample(
                len(mu_predicted), z=mu_predicted, temp=2
            )
            predicted_smiles_variants[j].extend(predicted_smiles_batch_j)

    predicted_dict = {}
    for i in range(num_variants):
        predicted_dict["pred_sample_" + str(i)] = predicted_smiles_variants[i]
    predicted_smiles_list = merge_lists(predicted_dict)
    final_dict = {
        "smiles_with_precursor": smiles_with_precursor,
        "predicted_smiles_list": predicted_smiles_list,
    }

    samples = pd.DataFrame(final_dict)

    print(samples.head())
    samples.to_csv(output_file, index=False, header=True)


def merge_lists(predicted_dict):
    keys = list(predicted_dict.keys())
    num_mols = len(predicted_dict[keys[0]])
    merged = []
    for i in range(num_mols):
        merged.append("|".join(set([predicted_dict[k][i] for k in keys])))
    return merged


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="A file containing a list of smiles",
    )

    parser.add_argument(
        "--predicted_embeddings",
        type=str,
        required=True,
        help="Embeddings created from the spectrum",
    )

    parser.add_argument(
        "--model_load",
        type=str,
        required=True,
        help="Where to load the model",
    )
    parser.add_argument(
        "--config_load",
        type=str,
        required=True,
        help="Where to load the config",
    )

    parser.add_argument(
        "--vocab_load",
        type=str,
        required=True,
        help="Where to load the vocab",
    )
    parser.add_argument(
        "--device", type=str, required=True, help="cuda/cpu?",
    )

    parser.add_argument(
        "--n_batch", type=int, required=True, help="Batch Size",
    )

    parser.add_argument(
        "--num_variants", type=int, required=True, help="Number of Variants",
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")

    return parser


def main(model, config):
    #debugpy()
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)
    MODELS = ModelsStorage()

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    print(model_vocab.c2i)
    model_state = torch.load(config.model_load)

    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    predicted_embeddings = torch.load(config.predicted_embeddings)

    smiles_keys = list(predicted_embeddings.keys())

    data = [smiles_keys]

    dataset = ListDataset(data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.n_batch, shuffle=False
    )

    decode(
        model,
        train_loader,
        predicted_embeddings,
        config.output_file,
        config.num_variants,
    )


if __name__ == "__main__":
    parser = get_parser()
    print(parser)
    config = parser.parse_args()
    print(config)
    # model = sys.argv[1]
    main(config.model, config)
