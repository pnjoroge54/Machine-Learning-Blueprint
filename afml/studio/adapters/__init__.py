"""
Adapters between the studio core and afml.

``pipelines`` is the one module that knows which afml functions back each studio
stage. When the newer afml (FeatureEngineeringPipeline, ModelDevelopmentPipeline,
screen_features, ...) lands, rewire that module and nothing else.
"""
