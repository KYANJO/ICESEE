# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-17
# @description: Add or remove models to be supported by the application here.
#               Currently supported models include:
#               - icepack
#               - Lorenz96
#               - flowline (integration still underway)
#               - ISSM    (development still underway)
# =============================================================================

# --- Imports ---
import sys
import os
import re
import importlib

class SupportedModels:
    """
    Class to call the supported models in the application.
    Easily add/remove models by updating MODEL_CONFIG dictionary
    """

    # Dictionary mapping model names to their respective import paths and states
    MODEL_CONFIG = {
        "icepack": {
            "module": "icepack_model.icepack_enkf",
            "description": "Icepack model",
            "status": "supported",
        },
        "lorenz96": {
            "module": "lorenz96.lorenz96_enkf",
            "description": "Lorenz96 model",
            "status": "supported",
        },
        "flowline": {
            "module": "flowline.flowline_enkf",
            "description": "Flowline model",
            "status": "development",
        },
        "issm": {
            "module": "issm.issm_enkf",
            "description": "ISSM model",
            "status": "development",
        },
    }

    def __init__(self, model=None, model_config=None):
        self.model = model
        self.MODEL_CONFIG = model_config or self.MODEL_CONFIG

    def _get_project_root():
        """Automatically determines the root of the project."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Traverse upwards until we reach the root of the project (assuming 'src' folder exists at root)
        while not os.path.exists(os.path.join(current_dir, 'src')):
            current_dir = os.path.dirname(current_dir)
        return current_dir
    
    # Now globally add application directory to the path
    project_root = _get_project_root()
    application_dir = os.path.join(project_root, 'applications')
    sys.path.insert(0, application_dir)

    def list_models(self):
        """List all supported models with their descriptions and statuses."""
        for model, info in self.MODEL_CONFIG.items():
            print(f"{model.capitalize()}: {info['description']} (Status: {info['status']})")

    def call_model(self):
        """
        Dynamically import and return the modules for the specified model.
        """
        if not self.model:
            raise ValueError("No model specified. Please provide a model name.")

        # Normalize model name for case-insensitive matching
        normalized_model = self.model.lower()

        if normalized_model not in self.MODEL_CONFIG:
            raise ValueError(f"Model '{self.model}' is not supported or implemented.")

        model_info = self.MODEL_CONFIG[normalized_model]

        # Check model status
        if model_info["status"] != "supported":
            raise ValueError(f"Model '{self.model}' is still under development: {model_info['description']}.")

        try:
            # Dynamically import the model 
            model_module = importlib.import_module(model_info["module"])
            print(f"Successfully loaded {model_info['description']} from {model_info['module']}.")
            return model_module
        except ImportError as e:
            raise ImportError(f"Failed to import module for model '{self.model}': {e}")
