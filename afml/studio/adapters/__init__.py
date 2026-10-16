"""
Adapters between the studio core and afml.

``pipelines`` is the one module that knows which afml functions back each studio
stage. Moving onto FeatureEngineeringPipeline, ModelDevelopmentPipeline,
screen_features and the rest of the newer afml means rewiring that module and
nothing else.
"""
