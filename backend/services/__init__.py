def __init__(self, pipeline_path: str = None) -> None:
        # 1. D'abord on définit les chemins et on charge le modèle
        self._project_root = Path(__file__).resolve().parent.parent.parent
        self._preprocessor = self._load_preprocessing_pipeline()
        self._model = self._load_model()
        self._vt = self._load_variance_threshold()
        
        # 2. SEULEMENT À LA FIN, on charge les noms des features
        self._feature_names = self._load_feature_names()