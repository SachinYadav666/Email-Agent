"""Email Agent Utilities

This module implements an email processing workflow using LangGraph and LangChain.
It provides functionality for categorizing emails, conducting research, drafting responses,
and refining the content based on analysis.

The workflow follows these main steps:
1. Email categorization
2. Research routing and information gathering
3. Draft email generation
4. Analysis and potential rewriting
5. Final email production

The system uses Groq LLM for processing and Tavily for web searches.
"""
from langchain_tavily import TavilySearch
from langchain.schema import Document
from typing import List, TypedDict, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from django.conf import settings
from .prompt_utils import (
    EMAIL_CATEGORY_PROMPT,
    RESEARCH_ROUTER_PROMPT,
    SEARCH_KEYWORD_PROMPT,
    DRAFT_WRITER_PROMPT,
    REWRITE_ROUTER_PROMPT,
    DRAFT_ANALYSIS_PROMPT
)
import logging
from datetime import datetime

# Initialize logging
logger = logging.getLogger(__name__)

# Constants
MAX_SEARCH_ATTEMPTS = 1
MAX_WORKFLOW_STEPS = 5

# Initialize LLM
GROQ_LLM = ChatGroq(
    model=settings.LLM_MODEL_NAME,
    api_key=settings.GROQ_API_KEY
)

# Initialize Tavily search
tavily_search = TavilySearch(api_key=settings.TAVILY_API_KEY)

# State definition
class GraphState(TypedDict):
    """
    Represents the state of our email processing graph.

    Attributes:
        initial_email (str): The original email content to be processed
        email_category (str): Category determined for the email (e.g., 'inquiry', 'complaint')
        draft_email (str): The current draft version of the email
        final_email (str): The final version of the email after processing
        research_info (List[str]): List of research documents gathered during processing
        info_needed (bool): Flag indicating if additional research is needed
        num_steps (int): Counter for tracking workflow steps to prevent infinite loops
        draft_email_feedback (dict): Analysis feedback for the current draft
    """
    initial_email: str
    email_category: str
    draft_email: str
    final_email: str
    research_info: List[str]
    info_needed: bool
    num_steps: int
    draft_email_feedback: dict

# Chain definitions
email_category_generator = EMAIL_CATEGORY_PROMPT | GROQ_LLM | StrOutputParser()
research_router = RESEARCH_ROUTER_PROMPT | GROQ_LLM | JsonOutputParser()
search_keyword_chain = SEARCH_KEYWORD_PROMPT | GROQ_LLM | JsonOutputParser()
draft_writer_chain = DRAFT_WRITER_PROMPT | GROQ_LLM | JsonOutputParser()
rewrite_router = REWRITE_ROUTER_PROMPT | GROQ_LLM | JsonOutputParser()
draft_analysis_chain = DRAFT_ANALYSIS_PROMPT | GROQ_LLM | JsonOutputParser()

# Utility functions
def write_markdown_file(content: str, filename: Optional[str] = None) -> str:
    """Writes the given content as a markdown file with timestamp.

    Args:
        content: The string content to write to the file
        filename: Optional custom filename. If not provided, generates timestamp-based name

    Returns:
        str: The path to the written file

    Raises:
        IOError: If there are issues writing to the file
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"email_draft_{timestamp}"
        
        filepath = f"{filename}.md"
        with open(filepath, "w") as f:
            f.write(content)
        logger.info(f"Successfully wrote markdown file: {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Failed to write markdown file: {str(e)}")
        raise

# Graph node functions
def categorize_email(state: GraphState) -> GraphState:
    """Categorize the email using the LLM."""
    email_category = email_category_generator.invoke({"initial_email": state["initial_email"]})
    return {**state, "email_category": email_category}

def research_info_search(state: GraphState) -> GraphState:
    """Search for relevant information based on email content and category.
    
    This function performs web searches using extracted keywords to gather
    relevant information for drafting the email response. It includes error
    handling and retry logic for failed searches.

    Args:
        state: Current graph state containing email and category information

    Returns:
        Updated graph state with research information and step count

    Raises:
        ValueError: If the state is invalid or search fails after retries
    """
    logger.info("Starting research info search")
    
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    research_info = state["research_info"]
    num_steps = state['num_steps']
    num_steps += 1

    try:
        # Extract search keywords
        keywords = search_keyword_chain.invoke({
            "initial_email": initial_email,
            "email_category": email_category
        })
        keywords = keywords['keywords']
        
        if not keywords:
            logger.warning("No keywords generated for search")
            return {**state, "research_info": research_info, "num_steps": num_steps}

        full_searches = []
        search_attempts = 0
        
        for keyword in keywords[:1]:  # Limit to first keyword for efficiency
            while search_attempts < MAX_SEARCH_ATTEMPTS:
                try:
                    logger.info(f"Searching for keyword: {keyword}")
                    # Use Tavily search directly
                    search_results = tavily_search.invoke(keyword)
                    
                    if not search_results:
                        logger.warning(f"No results found for keyword: {keyword}")
                        break
                        
                    # Process search results
                    for result in search_results:
                        content = f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}\nURL: {result.get('url', '')}"
                        doc = Document(page_content=content)
                        full_searches.append(doc)
                    
                    logger.info(f"Found {len(full_searches)} search results")
                    break
                    
                except Exception as e:
                    search_attempts += 1
                    logger.error(f"Search attempt {search_attempts} failed: {str(e)}")
                    if search_attempts == MAX_SEARCH_ATTEMPTS:
                        logger.error("Max search attempts reached, proceeding with available information")
                        break

        # Log the research results
        if full_searches:
            logger.info(f"Successfully gathered {len(full_searches)} research documents")
            # Optionally save research to markdown for debugging
            if settings.DEBUG:
                write_markdown_file(
                    "\n\n".join([doc.page_content for doc in full_searches]),
                    f"research_info_step_{num_steps}"
                )
        else:
            logger.warning("No research information gathered")

        return {
            **state,
            "research_info": full_searches if full_searches else research_info,
            "num_steps": num_steps
        }

    except Exception as e:
        logger.error(f"Research search failed: {str(e)}")
        raise ValueError(f"Failed to complete research search: {str(e)}")

def draft_email_writer(state: GraphState) -> GraphState:
    """Write a draft email based on the category and research information.
    
    This function generates an email draft using the LLM, incorporating
    the original email content, determined category, and any gathered research.
    It includes step counting to prevent infinite loops.

    Args:
        state: Current graph state containing email, category, and research information

    Returns:
        Updated graph state with the new draft email

    Raises:
        ValueError: If the state is invalid or draft generation fails
    """
    if state["num_steps"] >= MAX_WORKFLOW_STEPS:
        logger.warning("Maximum workflow steps reached, using current draft")
        return state

    try:
        draft = draft_writer_chain.invoke({
            "initial_email": state["initial_email"],
            "email_category": state["email_category"],
            "research_info": state["research_info"]
        })
        
        # Save draft for debugging if in debug mode
        if settings.DEBUG:
            write_markdown_file(
                draft["email_draft"],
                f"draft_email_step_{state['num_steps']}"
            )
            
        return {**state, "draft_email": draft["email_draft"]}
    except Exception as e:
        logger.error(f"Draft generation failed: {str(e)}")
        raise ValueError(f"Failed to generate email draft: {str(e)}")

def analyze_draft_email(state: GraphState) -> GraphState:
    """Analyze the draft email for quality and completeness."""
    try:
        # If we already have a draft email, use it as final
        if state["draft_email"] and not state["final_email"]:
            return {**state, "final_email": state["draft_email"]}
            
        analysis = draft_analysis_chain.invoke({
            "initial_email": state["initial_email"],
            "email_category": state["email_category"],
            "research_info": state["research_info"],
            "draft_email": state["draft_email"]
        })
        return {**state, "draft_email_feedback": analysis}
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        # On any error, use current draft as final
        return {**state, "final_email": state["draft_email"]}

def rewrite_email(state: GraphState) -> GraphState:
    """Rewrite the email based on feedback."""
    new_draft = draft_writer_chain.invoke({
        "initial_email": state["initial_email"],
        "email_category": state["email_category"],
        "research_info": state["research_info"]
    })
    return {**state, "draft_email": new_draft["email_draft"]}

def no_rewrite(state: GraphState) -> GraphState:
    """Keep the current draft as final."""
    return {**state, "final_email": state["draft_email"]}

def state_printer(state: GraphState) -> GraphState:
    """Print the current state for debugging."""
    print(f"Current state: {state}")
    return state

def route_to_research(state: GraphState) -> Dict[str, str]:
    """Route decision for research vs direct draft."""
    decision = research_router.invoke({
        "initial_email": state["initial_email"],
        "email_category": state["email_category"]
    })
    return {"next": decision["router_decision"]}

def route_to_rewrite(state: GraphState) -> Dict[str, str]:
    """Route decision for rewriting vs keeping draft."""
    try:
        # If we already have a final email, end the workflow
        if state["final_email"]:
            return {"next": "no_rewrite"}
            
        # If we have a draft but no analysis, use the draft
        if state["draft_email"] and not state["draft_email_feedback"]:
            return {"next": "no_rewrite"}
            
        decision = rewrite_router.invoke({
            "initial_email": state["initial_email"],
            "email_category": state["email_category"],
            "draft_email": state["draft_email"]
        })
        return {"next": decision["router_decision"]}
    except Exception as e:
        logger.error(f"Rewrite routing failed: {str(e)}")
        # On any error, use current draft
        return {"next": "no_rewrite"}

def create_email_agent_graph():
    """Create the email agent workflow graph.
    
    This function constructs a directed graph representing the email processing workflow.
    The graph includes nodes for categorization, research, drafting, analysis, and rewriting.
    It implements conditional routing based on the needs of each email.
    
    The workflow follows these steps:
    1. Categorize the email
    2. Determine if research is needed
    3. Gather research if required
    4. Generate initial draft
    5. Analyze draft quality
    6. Rewrite if necessary
    7. Produce final email
    
    Returns:
        A compiled StateGraph ready for execution
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("categorize", categorize_email)
    workflow.add_node("research", research_info_search)
    workflow.add_node("draft", draft_email_writer)
    workflow.add_node("analyze", analyze_draft_email)
    workflow.add_node("rewrite", rewrite_email)
    workflow.add_node("no_rewrite", no_rewrite)
    workflow.add_node("print_state", state_printer)
    workflow.add_node("route_research", route_to_research)
    workflow.add_node("route_rewrite", route_to_rewrite)

    # Add edges with proper end conditions
    workflow.add_edge("categorize", "print_state")
    workflow.add_edge("print_state", "route_research")
    workflow.add_conditional_edges(
        "route_research",
        lambda x: x["next"],
        {
            "research_info": "research",
            "draft_email": "draft"
        }
    )
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", "analyze")
    workflow.add_edge("analyze", "route_rewrite")
    workflow.add_conditional_edges(
        "route_rewrite",
        lambda x: x["next"],
        {
            "rewrite": "rewrite",
            "no_rewrite": "no_rewrite"
        }
    )
    # Ensure rewrite goes back to analyze only if we don't have a final email
    workflow.add_conditional_edges(
        "rewrite",
        lambda x: "analyze" if not x.get("final_email") else "no_rewrite",
        {
            "analyze": "analyze",
            "no_rewrite": "no_rewrite"
        }
    )
    workflow.add_edge("no_rewrite", END)

    # Set entry point
    workflow.set_entry_point("categorize")

    # Compile the workflow
    compiled_workflow = workflow.compile()

   

    return compiled_workflow





# categorize
#    ↓
# print_state
#    ↓
# route_research ───> research ──┐
#      ↓                         ↓
#    draft <─────────────────────┘
#      ↓
#  analyze
#    ↓
# route_rewrite ──> rewrite ─────┐
#      ↓                         ↓
# no_rewrite <────── analyze <───┘
#    ↓
#   END
