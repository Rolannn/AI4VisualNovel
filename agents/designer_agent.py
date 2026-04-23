"""
Designer Agent
~~~~~~~~~~~~~~
策划 Agent - 负责草拟游戏整体设计文档
"""

import logging
from typing import Dict, Any, Optional
from .llm_client import LLMClient
import json

from .config import DesignerConfig, PathConfig
from .utils import JSONParser, FileHelper

logger = logging.getLogger(__name__)


class DesignerAgent:
    """策划 Agent - 游戏设计文档草拟者"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化策划 Agent
        """
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)
        self.config = DesignerConfig
        
        logger.info("✅ 策划 Agent 初始化成功")
    
    def generate_game_design(
        self,
        character_count: int = None,
        requirements: str = "",
        feedback: str = None,
        previous_game_design: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        生成或修改游戏设计
        
        Args:
            character_count: 角色数量
            requirements: 用户需求
            feedback: 制作人反馈 (可选，用于优化模式)
            previous_game_design: 之前生成的游戏设计 (可选，用于优化模式)
        """
        character_count = character_count or self.config.DEFAULT_CHARACTER_COUNT
        
        logger.info("📝 策划正在生成游戏设计...")
        
        try:
            # 构建基础 prompt
            user_prompt = self.config.GAME_DESIGN_PROMPT.format(
                character_count=character_count,
                total_nodes=self.config.TOTAL_NODES,
                requirements=requirements if requirements else "无"
            )
            
            # 如果有反馈和之前的设计，直接追加
            if feedback and previous_game_design:
                logger.info("🔧 优化模式：根据反馈修改...")
                user_prompt += f"\n\n【原游戏设计】\n{json.dumps(previous_game_design, ensure_ascii=False, indent=2)}\n\n【制作人反馈】\n{feedback}\n\n请修改游戏设计文档，解决制作人提出的问题。保持 JSON 格式不变，只修改内容。"
            
            content = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.TEMPERATURE,
                json_mode=True
            )
            
            game_design = JSONParser.parse_ai_response(content)
            
            # 验证必要字段
            required_fields = ["title", "background", "story_graph", "characters", "scenes"]
            if not JSONParser.validate_required_fields(game_design, required_fields):
                raise ValueError("生成的设计文档缺少必需字段")

            # 节点数量检查：如果 LLM 生成超出限制，记录警告
            sg = game_design.get("story_graph", {})
            raw_nodes = sg.get("nodes", {})
            actual_count = len(raw_nodes) if isinstance(raw_nodes, dict) else len(raw_nodes)
            target = self.config.TOTAL_NODES
            if actual_count > target + 1:
                logger.warning(
                    f"⚠️  LLM 生成了 {actual_count} 个节点，超出目标 {target}±1。"
                    f" 建议在 .env 中调整 GAME_TOTAL_NODES 或重新生成。"
                )
            else:
                logger.info(f"✅ 节点数量符合要求: {actual_count}/{target}")

            logger.info(f"✅ 游戏设计完成: 《{game_design['title']}》")
            return game_design
            
        except Exception as e:
            logger.error(f"❌ 游戏设计生成失败: {e}")
            raise
