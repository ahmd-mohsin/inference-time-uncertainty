import logging

from src.search.rmi_tree_search import RMITreeSearchGenerator, RMITreeResult

logger = logging.getLogger(__name__)

class REBASEGenerator(RMITreeSearchGenerator):

    def __init__(self, model, tokenizer, prm, cfg: dict):
        cfg_copy = {**cfg}
        search = {**cfg_copy.get("search", {})}
        search["lambda_diversity"] = 0.0
        cfg_copy["search"] = search

        super().__init__(model, tokenizer, prm, cfg_copy)
        logger.info("REBASE baseline initialized (λ=0, PRM-only scoring)")