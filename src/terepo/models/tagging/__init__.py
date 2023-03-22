from transformers import AutoConfig, AutoFeatureExtractor

from .gector.configuration_gector import GECToRConfig
from .gector.modeling_gector import GECToRModel
from .gector.feature_extraction_gector import GECToRFeatureExtractor

AutoConfig.register('gector', GECToRConfig)
AutoFeatureExtractor.register('gector', GECToRFeatureExtractor)
