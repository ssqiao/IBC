import settings
from loader.model_loader import load_model
from feature_operation import hook_feature, FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean

fo = FeatureOperator()  # construct broden seg. data loader and prefetcher
model = load_model(hook_feature)  # load pre-trained model and target hooked conv layer

# STEP 1: feature extraction
features, maxfeature = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    # STEP 2: calculating threshold
    thresholds = fo.quantile_threshold(features[layer_id], savepath="quantile.npy")

    # STEP 3: calculating IoU scores
    tally_result = fo.tally(features[layer_id], thresholds, savepath="tally.csv")

    # STEP 4: generating results
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()  # qss clean the temporary large files
