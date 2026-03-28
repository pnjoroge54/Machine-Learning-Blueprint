"""
Enhanced implementation using separate bid/ask prices for long/short models.

This approach aligns training data with actual execution prices, improving
model performance in production trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger


class BidAskLongShortPipeline:
    """
    Enhanced pipeline using realistic bid/ask prices for each side.
    
    This approach provides better production performance by training models
    on the actual prices they'll encounter during execution.
    """
    
    def __init__(
        self,
        strategy,
        data_config: dict,
        feature_config: dict,
        target_config: dict,
        label_config: dict,
        model_params: dict,
        base_dir: str = "Models/BidAsk_LongShort",
    ):
        """
        Initialize bid/ask-aware long/short pipeline.
        """
        from .model_development import ModelDevelopmentPipeline
        
        self.strategy = strategy
        self.data_config = data_config
        self.feature_config = feature_config
        self.target_config = target_config
        self.label_config = label_config
        self.model_params = model_params
        
        # Will be populated during run
        self.bar_data = None
        self.long_bars = None
        self.short_bars = None
        self.spread_series = None
        
        # Create configurations for each side
        self.long_config = data_config.copy()
        self.long_config['price'] = 'ask'  # Long positions use ask
        self.long_config['model_name'] = f"{data_config.get('symbol', 'MODEL')}_LONG_ASK"
        
        self.short_config = data_config.copy()
        self.short_config['price'] = 'bid'  # Short positions use bid
        self.short_config['model_name'] = f"{data_config.get('symbol', 'MODEL')}_SHORT_BID"
        
        self.long_pipeline = ModelDevelopmentPipeline(
            strategy=strategy,
            data_config=self.long_config,
            feature_config=feature_config,
            target_config=target_config,
            label_config=label_config,
            model_params=model_params.copy(),
            base_dir=f"{base_dir}/Long_Ask"
        )
        
        self.short_pipeline = ModelDevelopmentPipeline(
            strategy=strategy,
            data_config=self.short_config,
            feature_config=feature_config,
            target_config=target_config,
            label_config=label_config,
            model_params=model_params.copy(),
            base_dir=f"{base_dir}/Short_Bid"
        )
        
    def run(
        self,
        generate_reports: bool = True,
        save: bool = True,
        export_onnx: bool = False,
        calibrate: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Run bid/ask-aware long/short model development pipeline.
        """
        self.export_onnx = export_onnx
        
        if verbose:
            print("\n" + "=" * 80)
            print("BID/ASK-AWARE LONG/SHORT MODEL DEVELOPMENT PIPELINE")
            print("=" * 80)
            
        try:       
            # Step 1: Create separate bars for long (ask) and short (bid)
            if verbose:
                print("\n[Step 1/6] Creating side-specific bars...")
            self._create_side_specific_bars()
            
            # Step 2: Engineer features for each side
            if verbose:
                print("\n[Step 2/6] Engineering features for each side...")
            self._engineer_side_specific_features()
            
            # Step 3: Generate events for each side
            if verbose:
                print("\n[Step 3/6] Generating side-specific events...")
            self._generate_side_specific_events()
            
            # Step 4: Train long model
            if verbose:
                print("\n[Step 4/6] Training LONG model (ASK-based)...")
            long_results = self._train_side_model(
                self.long_pipeline,
                "LONG (ASK)",
                generate_reports,
                save,
                export_onnx,
                calibrate,
                verbose
            )
            
            # Step 5: Train short model
            if verbose:
                print("\n[Step 5/6] Training SHORT model (BID-based)...")
            short_results = self._train_side_model(
                self.short_pipeline,
                "SHORT (BID)",
                generate_reports,
                save,
                export_onnx,
                calibrate,
                verbose
            )
            
            # Step 6: Generate spread-aware analysis
            if verbose:
                print("\n[Step 6/6] Generating spread-aware analysis...")
            combined_metrics = self._generate_spread_analysis(long_results, short_results)
            
            results = {
                'long_model': long_results[0],
                'short_model': short_results[0],
                'long_features': long_results[1],
                'short_features': short_results[1],
                'long_metrics': long_results[2],
                'short_metrics': short_results[2],
                'combined_metrics': combined_metrics,
                'spread_stats': self._calculate_spread_statistics(),
                'long_config': long_results[3],
                'short_config': short_results[3],
            }
            
            if verbose:
                print("\n" + "=" * 80)
                print("✓ Bid/Ask-Aware Pipeline Completed Successfully")
                print("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Bid/Ask pipeline failed: {e}")
            raise
        
    def _create_side_specific_bars(self):
        """Create separate bars using ask (long) and bid (short) prices."""
        from .model_development import load_and_prepare_training_data
        
        fetch_config = self.data_config.copy()
        fetch_config['price'] = 'bid_ask'
        
        self.bar_data = load_and_prepare_training_data(**fetch_config)
        
        if self.data_config['bar_type'] == "tick":
            bar_size = self.bar_data["tick_volume"].iloc[0]
            self.short_config["tick_bar_size"] = bar_size
            self.long_config["tick_bar_size"] = bar_size
            
        self.long_bars = self.bar_data.filter(regex='ask').copy()
        self.long_bars.columns = [x.split('_')[1] for x in self.long_bars.columns]
        
        self.short_bars = self.bar_data.filter(regex='bid').copy()
        self.short_bars.columns = [x.split('_')[1] for x in self.short_bars.columns]
        
        self.spread_series = self.long_bars['close'] - self.short_bars['close']
        self.short_bars['spread'] = self.spread_series
        self.long_bars['spread'] = self.spread_series
        
    def _engineer_side_specific_features(self):
        """Engineer features separately for each side."""
        from .model_development import create_feature_engineering_pipeline
        
        long_features = create_feature_engineering_pipeline(
            self.long_bars,
            self.feature_config,
            self.long_config
        )
        
        short_features = create_feature_engineering_pipeline(
            self.short_bars,
            self.feature_config,
            self.short_config
        )
        
        self.long_pipeline.bar_data = self.long_bars
        self.short_pipeline.bar_data = self.short_bars
        self.long_pipeline.features = long_features
        self.short_pipeline.features = short_features
        
        self.long_pipeline.completed_steps['data_loading'] = True
        self.short_pipeline.completed_steps['data_loading'] = True
        self.long_pipeline.completed_steps['feature_engineering'] = True
        self.short_pipeline.completed_steps['feature_engineering'] = True
        
    def _generate_side_specific_events(self):
        """Generate events separately for each side using appropriate prices."""
        from .model_development import generate_events_triple_barrier
        
        long_events = generate_events_triple_barrier(
            self.long_bars,
            self.strategy,
            self.target_config,
            **self.label_config
        )
        long_events = long_events[long_events['side'] == 1]
        
        short_events = generate_events_triple_barrier(
            self.short_bars,
            self.strategy,
            self.target_config,
            **self.label_config
        )
        short_events = short_events[short_events['side'] == -1]
        
        self.long_pipeline.events = long_events
        self.short_pipeline.events = short_events
        
        self.long_pipeline.completed_steps['label_generation'] = True
        self.short_pipeline.completed_steps['label_generation'] = True
        
    def _train_side_model(self, pipeline, side_name, generate_reports, save, export_onnx, verbose):
        """Train model for specific side."""
        pipeline.compute_sample_weights()
        pipeline.add_meta_features()
        pipeline.preprocess_features()
        pipeline.train_model()
        pipeline.analyze_features()
        pipeline._compile_metrics()
        pipeline.export_onnx = self.export_onnx
        
        if generate_reports:
            pipeline._generate_analysis_reports()
        if save:
            pipeline._save_all_artifacts()
            
        return (
            pipeline.best_model,
            pipeline._get_feature_names(),
            pipeline.metrics,
            pipeline.config
        )
    
    def _calculate_spread_statistics(self) -> Dict:
        """Calculate spread statistics at bar frequency."""
        mid_price = (self.long_bars['close'] + self.short_bars['close']) / 2
        spread_bps = float((self.spread_series.mean() / mid_price.mean()) * 10000)
        return {
            'spread_mean':   float(self.spread_series.mean()),
            'spread_std':    float(self.spread_series.std()),
            'spread_median': float(self.spread_series.median()),
            'spread_95th':   float(self.spread_series.quantile(0.95)),
            'spread_bps':    spread_bps,
        }
    
    def _generate_spread_analysis(self, long_results, short_results) -> Dict:
        """Generate analysis including spread impact."""
        return {
            'long_events':    long_results[2]['training_samples'],
            'short_events':   short_results[2]['training_samples'],
            'long_cv_score':  long_results[2]['cv_results'].get('best_score', 0),
            'short_cv_score': short_results[2]['cv_results'].get('best_score', 0),
            'long_features':  long_results[2]['feature_count'],
            'short_features': short_results[2]['feature_count'],
            'spread_stats':   self._calculate_spread_statistics(),
        }
