from sentence_transformers import SentenceTransformer
from settings import settings
from logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """
    Wraps SentenceTransformer with primary/fallback model loading.
    Lazy initialization: model loads on first call to encode().
    """
    def __init__(
            self,
            primary_model: str = settings.PRIMARY_EMBED_MODEL,
            fallback_model: str = settings.FALLBACK_EMBED_MODEL,
    ) -> None:
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        try:
            logger.info(f"Loading embedding model: {self._primary_model}")
            return SentenceTransformer(self._primary_model)
        except Exception as e:
            logger.warning(f"Failed to load primary model: {e}")
            logger.info(f"Loading fallback embedding model: {self._fallback_model}")
            return SentenceTransformer(self._fallback_model)
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._load()
        return self._model
    
    def encode(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()