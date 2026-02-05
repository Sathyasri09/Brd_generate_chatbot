from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Define the state of the agent
class AgentState(TypedDict):
    messages: List[Dict[str, str]] # Chat history
    user_goal: str  # The high-level goal of the user (refined)
    gathered_info: Dict[str, str] # Structure: { "section_name": "content" }
    missing_info: List[str] # List of sections that still need to be filled
    draft_html: Optional[str] # The generated HTML content
    ts_html: Optional[str] # The generated Technical Specification content
    version: int # To track document updates and force UI refresh
    conversation_active: bool # If false, the process is done

# Pydantic model for the "Refiner" output
class RefinedGoal(BaseModel):
    refined_goal: str = Field(default="", description="A clear, concise summary of the user's business need.")
    is_clear: bool = Field(default=False, description="True if the user's request is clear enough to start gathering requirements.")

# Pydantic model for the "Gatherer" output
class GathererOutput(BaseModel):
    new_info: Dict[str, Any] = Field(default_factory=dict, description="Information extracted from the last user message, mapped to BRD sections.")
    next_questions: List[str] = Field(default_factory=list, description="Specific follow-up questions to ask the user.")
    is_complete: bool = Field(default=False, description="True if all necessary information for the BRD has been gathered.")
    should_stop: bool = Field(default=False, description="True if the user explicitly asked to stop and generate the document now.")

