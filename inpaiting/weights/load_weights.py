from tensorflow import keras
from typing import List


def find_weights(
    weights_collection: dict, model_name: str, dataset: str, include_top: bool
) -> List[str]:
    """Called to find the correct weight correspoding
        to the model parameters.

    Args:
        weights_collection (dict): weights information;
        model_name (str): model name;
        dataset (str): dataset of the pre-trained model;
        include_top (bool): a bool to enable to use or not
            the top inclusion.

    Returns:
        List[str]: a list with the necessary informations to
            load the weights.
    """
    w = list(filter(lambda x: x["model"] == model_name, weights_collection))
    w = list(filter(lambda x: x["dataset"] == dataset, w))
    w = list(filter(lambda x: x["include_top"] == include_top, w))
    return w


def load_model_weights(
    weights_collection: dict,
    model: keras.Model,
    classes: int,
    include_top: bool,
    model_name: str,
    dataset: str = "imagenet",
) -> keras.Model:
    """Called to load ResNet-18 weights.

    Args:
        weights_collection (dict): weights info;
        model (keras.Model): the resnet model;
        classes (int): number of classes;
        include_top (bool): a bool to enable to use or not
            the top inclusion.
        dataset (str): dataset name of the pre-trained model;

    Returns:
        keras.Model: a ResNet-18 model with pre-trained weights.
    """

    weights = find_weights(weights_collection, model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights["classes"] != classes:
            raise ValueError(
                "If using `weights` and `include_top`"
                " as true, `classes` should be {}".format(weights["classes"])
            )

        weights_path = keras.utils.get_file(
            weights["name"], weights["url"], cache_subdir="models", md5_hash=weights["md5"]
        )

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    else:
        raise ValueError(
            "There is no weights for such configuration: "
            + "model = {}, dataset = {}, ".format(model.name, dataset)
            + "classes = {}, include_top = {}.".format(classes, include_top)
        )
    return model
