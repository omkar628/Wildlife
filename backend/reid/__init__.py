from backend.reid.interface import ReIDMatch, ReIDResult, TigerReIDBackend
from backend.reid.adapter import UnavailableReIDAdapter, inspect_reid_assets
from backend.reid.gallery import LocalTigerGallery
from backend.reid.identity import LocalIdentityService
from backend.reid.megadescriptor import MegaDescriptorEncoder, l2_normalize, select_device
from backend.reid.matching import (
    decide_match,
    decode_embedding,
    is_atrw_numeric_id,
    is_local_tiger_id,
)

__all__ = [
    "ReIDMatch",
    "ReIDResult",
    "TigerReIDBackend",
    "UnavailableReIDAdapter",
    "inspect_reid_assets",
    "LocalTigerGallery",
    "LocalIdentityService",
    "MegaDescriptorEncoder",
    "l2_normalize",
    "select_device",
    "decide_match",
    "decode_embedding",
    "is_atrw_numeric_id",
    "is_local_tiger_id",
]
