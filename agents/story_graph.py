"""
Story Graph Utilities
~~~~~~~~~~~~~~~~~~~~~
Helpers for directed acyclic story graphs (tree / DAG).
"""

from typing import Dict, List, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class StoryGraph:
    """Story graph (tree or DAG) from `story_graph` in the game design."""

    def __init__(self, data: Dict[str, Any]):
        """
        Args:
            data: Game design dict; must include `story_graph`
        """
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        self.reverse_adjacency = {}

        if 'story_graph' not in data:
            raise ValueError("game design is missing the story_graph field")

        self._load_dag_format(data['story_graph'])

    def _load_dag_format(self, story_graph: Dict[str, Any]):
        self.nodes = story_graph.get('nodes', {})
        self.edges = story_graph.get('edges', [])

        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.reverse_adjacency = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            from_node = edge['from']
            to_node = edge['to']
            choice_text = edge.get('choice_text')

            self.adjacency[from_node].append((to_node, choice_text))
            self.reverse_adjacency[to_node].append(from_node)

        logger.info(f"Loaded story graph: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def get_children(self, node_id: str) -> List[Tuple[str, str]]:
        return self.adjacency.get(node_id, [])

    def get_parents(self, node_id: str) -> List[str]:
        return self.reverse_adjacency.get(node_id, [])

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id, {})

    def is_merge_point(self, node_id: str) -> bool:
        return len(self.get_parents(node_id)) > 1

    def topological_sort(self) -> List[str]:
        in_degree = {node_id: len(self.get_parents(node_id)) for node_id in self.nodes}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for child_id, _ in self.get_children(node_id):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(result) != len(self.nodes):
            logger.error("Graph has a cycle; topological sort failed")
            return []

        return result

    def validate(self) -> Tuple[bool, str]:
        for edge in self.edges:
            if edge['from'] not in self.nodes:
                return False, f"Edge references missing source node: {edge['from']}"
            if edge['to'] not in self.nodes:
                return False, f"Edge references missing target node: {edge['to']}"

        sorted_nodes = self.topological_sort()
        if not sorted_nodes:
            return False, "Graph contains a cycle"

        if 'root' not in self.nodes:
            return False, "Missing root start node"

        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'edges': self.edges
        }

    def get_reachable_endings(self, from_node: str) -> List[str]:
        visited = set()
        endings = []

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            children = self.get_children(node_id)
            if not children:
                endings.append(node_id)
            else:
                for child_id, _ in children:
                    dfs(child_id)

        dfs(from_node)
        return endings
