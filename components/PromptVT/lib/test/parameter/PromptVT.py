from ...test.utils import TrackerParams
import os
from ...test.evaluation.environment import env_settings
from ...config.PromptVT.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    # prj_dir = env_settings().prj_dir
    # save_dir = env_settings().save_dir
    # update default config from yaml file
    # yaml_file = os.path.join(prj_dir, 'experiments/PromptVT/%s.yaml' % yaml_name)
    yaml_file = r"D:\down\Pattern_Recognition_System-main\components\PromptVT\experiments\PromptVT\baseline.yaml"
    update_config_from_file(yaml_file)
    params.cfg = cfg
    #print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE


    params.checkpoint ="../checkpoints/PromptVT/baseline/PromptVT.pth"


    params.save_all_boxes = False
    return params
