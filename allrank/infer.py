import torch
from attr import asdict
import torch.nn as nn

from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, load_libsvm_dataset_role
from allrank.models.model import FCModel, LTRModel, OutputLayer
from allrank.models.transformer import make_transformer
from allrank.utils.file_utils import PathsContainer

id = "id"
# args = {}
class r:
    job_dir = ""
    run_id = ""
    config_file_name = ""

args = r()
args.job_dir = "D:\\Colecoes\\pasta-trabalho\\geoRiskListnetLossfold5-2-3\\"
args.run_id = id
args.config_file_name = "D:\\Colecoes\\pasta-trabalho\\geoRiskListnetLossfold5-2-3\\used_config.json"

paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

config = Config.from_json(paths.config_path)
input_path = "D:\\Colecoes\\BD\\web10k\\Fold5\\Norm.test.txt"
n_features = 136
slate_length = 1000
# test_ds = load_libsvm_dataset_role("test", input_path, slate_length)

def make_model(fc_model, transformer, post_model, n_features):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    if fc_model:
        fc_model = FCModel(**fc_model, n_features=n_features)  # type: ignore
    d_model = n_features if not fc_model else fc_model.output_size
    if transformer:
        transformer = make_transformer(n_features=d_model, **asdict(transformer, recurse=False))  # type: ignore
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, **post_model))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.load_state_dict(torch.load("D:\\Colecoes\\pasta-trabalho\\geoRiskListnetLossfold5-2-3\\model.pkl"))
    return model
# instantiate model
model = make_model(n_features=n_features, **asdict(config.model, recurse=False))

x = 2