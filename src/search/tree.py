import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class StepNode:

    node_id: int
    depth: int
    parent_id: Optional[int] = None

    token_ids: list[int] = field(default_factory=list)
    text: str = ""

    path_token_ids: list[int] = field(default_factory=list)
    path_text: str = ""

    prm_reward: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0

    continuation_topk_ids: list[int] = field(default_factory=list)
    continuation_topk_logprobs: list[float] = field(default_factory=list)

    is_terminal: bool = False
    is_complete: bool = False
    expansion_width: int = 0
    children_ids: list[int] = field(default_factory=list)

    extracted_answer: Optional[str] = None

@dataclass
class SolutionTree:

    nodes: dict[int, StepNode] = field(default_factory=dict)
    root_ids: list[int] = field(default_factory=list)
    _next_id: int = 0

    total_tokens_generated: int = 0
    n_prm_calls: int = 0
    n_completed_solutions: int = 0

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_node(self, node: StepNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id is not None and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.node_id)

    def get_leaves_at_depth(self, depth: int) -> list[StepNode]:
        return [
            n for n in self.nodes.values()
            if n.depth == depth and not n.is_terminal
        ]

    def get_terminal_nodes(self) -> list[StepNode]:
        return [n for n in self.nodes.values() if n.is_terminal]

    def get_complete_solutions(self) -> list[StepNode]:
        return [n for n in self.nodes.values() if n.is_complete]

    def get_path(self, node_id: int) -> list[StepNode]:
        path = []
        nid = node_id
        while nid is not None:
            node = self.nodes[nid]
            path.append(node)
            nid = node.parent_id
        return list(reversed(path))

    def max_depth(self) -> int:
        if not self.nodes:
            return -1
        return max(n.depth for n in self.nodes.values())