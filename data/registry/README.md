# Model Registry - Promoted offline models
# This directory contains *_latest.yaml files that describe promoted models
# The OnnxModelLoader will monitor this directory for changes every 60 seconds

# Example structure:
# es_classifier_latest.yaml
# nq_regressor_latest.yaml  
# regime_detector_latest.yaml

# Format for metadata files (YAML):
# version: "1.2.3+abcd1234"
# model_path: "models/es_classifier.v1.2.3+abcd1234.onnx"
# promoted_at: "2024-01-15T10:30:00Z"
# training_stats:
#   accuracy: 0.73
#   auc: 0.68
#   samples: 50000
# validation:
#   walk_forward_cv: true
#   out_of_sample_sharpe: 1.2