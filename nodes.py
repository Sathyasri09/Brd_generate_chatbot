import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from schema import AgentState, RefinedGoal, GathererOutput
from utils import load_template, parse_brd_sections, convert_html_to_docx

# Initialize LLM based on environment
llm_type = os.environ.get("LLM_TYPE", "google") 

if llm_type == "openrouter":
    # JSON LLM for structured output (Interview nodes)
    json_llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    # Document LLM for free-form output (BRD/TS generation)
    doc_llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2, # Slightly more creativity for writing
        max_tokens=8192
    )
else:
    json_llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-flash-latest"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0
    )
    doc_llm = json_llm # Gemini handles both well

TEMPLATE_PATH = "brd_template.txt"
BRD_TEMPLATE_CONTENT = load_template(TEMPLATE_PATH)
BRD_SECTIONS = parse_brd_sections(BRD_TEMPLATE_CONTENT)

def refine_node(state: AgentState) -> AgentState:
    """
    Principal Business Analyst: Establishes Project Mission.
    """
    messages = state['messages']
    last_message = messages[-1]['content']
    parser = PydanticOutputParser(pydantic_object=RefinedGoal)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Principal Business Analyst at a top-tier consulting firm. \n"
                   "Transform the user's initial request into a high-impact 'Project Mission Statement'. \n"
                   "Identify the Core Value, Primary Stakeholders, and measurable ROI. \n"
                   "If vague, ask ONE strategic question. If clear, output the refined mission statement. \n\n"
                   "OUTPUT INSTRUCTIONS: You MUST return a valid JSON object matching the schema. \n"
                   "{format_instructions}"),
        ("user", "{input}")
    ])
    
    chain = prompt | json_llm | parser
    
    try:
        result = chain.invoke({"input": last_message, "format_instructions": parser.get_format_instructions()})
        if result.is_clear:
            return {
                **state, 
                "user_goal": result.refined_goal, 
                "missing_info": BRD_SECTIONS.copy(), 
                "gathered_info": {},
                "version": 0
            }
        
        # If not clear, append the question to messages so it shows on UI
        return {
            **state, 
            "user_goal": "", 
            "messages": state['messages'] + [{"role": "assistant", "content": result.refined_goal}]
        }
    except Exception as e:
        print(f"Error in refiner: {e}")
        return state

def gather_info_node(state: AgentState) -> AgentState:
    """
    High-Efficiency Senior Business Analyst: Limits questioning to essential turns
    and avoids repeating questions.
    """
    if not state.get('user_goal'):
        return state

    parser = PydanticOutputParser(pydantic_object=GathererOutput)

    # Track which sections/questions have been asked
    asked_sections = state.get('asked_sections', set())

    # Only count assistant messages that are actual questions (we assume AI marks them)
    assistant_msgs = [m for m in state['messages'] if m['role'] == 'assistant']
    question_count = len(assistant_msgs)

    last_message = state['messages'][-1]['content']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Elite Senior Business Analyst performing a High-Quality gap analysis. \n"
                   "Project Goal: {goal}. \n"
                   "Previous Info: {gathered_info} \n\n"
                   "STRICT RULES:\n"
                   "1. One Question Only per turn.\n"
                   "2. Target the most critical missing detail.\n"
                   "3. Do NOT repeat a question already asked.\n"
                   "4. Stop if the user says 'Generate' or 'Enough'.\n"
                   "Current Turn: #{current_count}\n\n"
                   "OUTPUT INSTRUCTIONS: Return valid JSON matching schema with fields: "
                   "`new_info` (dict), `next_questions` (list), `is_complete` (bool), `asked_sections` (list)."),
        ("user", "{input}")
    ])

    chain = prompt | json_llm

    try:
        result = chain.invoke({
            "goal": state['user_goal'],
            "gathered_info": state.get('gathered_info', {}),
            "current_count": question_count + 1,
            "input": last_message,
            "asked_sections": list(asked_sections),
            "format_instructions": parser.get_format_instructions()
        })

        new_gathered = state.get('gathered_info', {}).copy()
        new_gathered.update(result.new_info)

        # Update asked sections so AI won't repeat
        new_asked_sections = asked_sections.union(set(result.asked_sections))

        # Determine completion
        user_forced_stop = any(word in last_message.upper() for word in ["GENERATE", "ENOUGH"])
        max_questions_reached = question_count >= 5
        is_complete = user_forced_stop or result.is_complete or max_questions_reached

        # Append next AI question if we are not complete
        new_messages = state['messages']
        if not is_complete and result.next_questions:
            next_question = result.next_questions[0]
            new_messages = state['messages'] + [{"role": "assistant", "content": next_question}]

        # Determine missing sections
        current_missing = [s for s in BRD_SECTIONS if s not in new_gathered]
        if not is_complete and not current_missing:
            current_missing = []  # No dummy placeholder, stop asking

        return {
            **state,
            "gathered_info": new_gathered,
            "missing_info": current_missing,
            "messages": new_messages,
            "asked_sections": new_asked_sections
        }

    except Exception as e:
        print(f"Error in gatherer: {e}")
        return state


def updater_node(state: AgentState) -> AgentState:
    """
    Expert Editor Agent: Updates requirements based on user feedback.
    """
    last_message = state['messages'][-1]['content']
    parser = PydanticOutputParser(pydantic_object=GathererOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Expert Executive Editor. Your objective is to extract NEW information from user feedback and merge it into the existing requirements. \n"
                   "Current Requirement Details: {gathered_info}. \n"
                   "User Feedback/Change Request: {input} \n\n"
                   "STRICT MERGING RULES:\n"
                   "1. **Extraction**: Identify precisely which BRD sections or subheadings are being addressed in the user message. \n"
                   "2. **Replacement**: If the user provides a direct update to an existing section, the NEW content must replace the OLD content in that section. \n"
                   "3. **Persistence**: Do NOT lose any existing information that was not requested to be changed. \n"
                   "4. **Formatting**: Ensure the output in `new_info` contains the updated content mapped to the section names. \n"
                   "{format_instructions}"),
        ("user", "{input}")
    ])
    
    chain = prompt | json_llm | parser
    
    try:
        result = chain.invoke({
            "gathered_info": state.get('gathered_info', {}),
            "input": last_message,
            "format_instructions": parser.get_format_instructions()
        })
        
        new_gathered = state.get('gathered_info', {}).copy()
        new_gathered.update(result.new_info)
        
        # Increment version to force app.py to detect a change
        current_version = state.get('version', 0)
        
        return {
            **state,
            "gathered_info": new_gathered,
            "version": current_version + 1,
            "conversation_active": False 
        }
    except Exception as e:
        print(f"Error in updater: {e}")
        return {
            **state,
            "messages": state['messages'] + [{"role": "assistant", "content": "I apologize, I encountered a technical error while processing that change. Could you please try rephrasing your request?"}]
        }

def generator_node(state: AgentState) -> AgentState:
    """
    Elite Technical Writer: Final Document Synthesis and Page Architecture.
    """
    info_to_use = state.get('gathered_info', {})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "SYSTEM PROMPT:\n"
                   "You are an Elite Senior Business Analyst and Technical Editor. \n"
                   "You must generate a comprehensive, professional Business Requirement Document (BRD). \n\n"
                   "**CRITICAL SYNTHESIS RULES (MANDATORY):**\n"
                   "1. **NO PLACEHOLDERS**: You are strictly forbidden from outputting template instructions. DELETE strings like '<Explain...>', '(to be filled by Business)', or notes in angle brackets/parentheses. \n"
                   "2. **PROFESSIONAL CONTENT**: If user input is missing for a section, use your project context (Project Goal) to DRAFT high-quality, professional requirements. Never leave a section empty or with generic instructions. \n"
                   "3. **CHECKBOXES**: Always mark checkboxes as [X] YES or [X] NO based on the project context (e.g., set Automation to [X] YES for automation projects). \n"
                   "4. **PAGE BREAK LIMIT**: Use the `<div class='page-break'></div>` marker EXACTLY TWICE: once after the Title Page, and once after the Table of Contents. Do NOT add it anywhere else. \n"
                   "5. **INCREMENTAL OVERRIDE**: Professional synthesis takes priority over previous drafts. If the previous draft contains placeholders or instructions, REPLACE them with actual content. \n"
                   "6. **TOTAL CAPTURE**: Every specific number (e.g. '80%'), name, or rule provided by the user MUST be included. \n"
                   "7. **DYNAMIC ARCHITECTURE**: Use headings (H1) and logical subheadings (H2, H3) for all 8 mandatory sections. \n"
                   "8. **CLEAN TABLE OF CONTENTS**: Page 2 MUST be a detailed Table of Contents. Formally increase the size of this section (e.g., wrap the list in a style with font-size: 14pt; font-weight: bold;). Use a bulleted list (<ul>) or plain text. Do NOT use ordered lists (<ol>), as headings already contain their own numbers (e.g., Use '1. Business Need' NOT '1. 1. Business Need'). \n\n"
                   "PROJECT GOAL (Mission): {project_goal} \n"
                   "Template Structure: {template_structure} \n"
                   "Previous Draft Context: {previous_draft} \n"
                   "Gathered Details / Changes: \n{gathered_info} \n\n"
                   "Output ONLY raw HTML. No markdown blocks."),
        ("user", "Generate the Final Elite BRD.")
    ])
    
    chain = prompt | doc_llm
    
    try:
        result = chain.invoke({
            "project_goal": state.get('user_goal', "Not provided"),
            "template_structure": BRD_TEMPLATE_CONTENT,
            "gathered_info": info_to_use,
            "previous_draft": state.get('draft_html', "No previous draft exists yet.")
        })
        
        content = result.content
        final_text = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'text' in item: final_text += item['text']
                elif hasattr(item, 'text'): final_text += item.text
                else: final_text += str(item)
            content = final_text

        html_content = content.replace('\\n', ' ').replace('\n', ' ').strip()
        # Robust cleanup for potential JSON leakage without breaking CSS
        if html_content.startswith('{') or '"brd"' in html_content[:100]:
            import re
            # Extract content from inside JSON quotes if it looks like the AI wrapped the HTML in JSON
            match = re.search(r'"(?:brd|content|html)":\s*"(.*)"\s*}?$', html_content, re.DOTALL)
            if match:
                html_content = match.group(1).replace('\\"', '"')
        
        # Aggressive cleaning of remaining artifacts
        html_content = html_content.replace('}{ "brd":', '').replace('}{ "content":', '').strip()
        if html_content.endswith('}'):
            html_content = html_content[:-1].strip()
        
        # Final safety to ensure no unescaped backslashes from JSON remain
        html_content = html_content.replace('\\r', '').replace('\\t', ' ')
        
        if html_content.startswith("```html"): html_content = html_content[7:-3].strip()
        elif html_content.startswith("```"): html_content = html_content[3:-3].strip()
        
        return {
            **state,
            "draft_html": html_content,
            "conversation_active": False
        }
    except Exception as e:
        print(f"Error in generator: {e}")
        return state

def ts_generator_node(state: AgentState) -> AgentState:
    """
    Technical Specification Agent: Generates developer-ready specs.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "SYSTEM PROMPT:\n"
                   "You are an Elite Technical Architect. \n"
                   "Based on the finalized BRD requirements, generate a professional Technical Specification Document (TSD). \n\n"
                   "TSD STRUCTURE:\n"
                   "1. Technical Overview: Architecture, System Components. \n"
                   "2. Data Model: Recommended SQL schema or Data structures. \n"
                   "3. Integration Logic: API specs, Error codes, Tolerances. \n"
                   "4. Security & Performance: Authentication, Data Privacy, Scalability. \n"
                   "5. Validation Logic: Detailed logic for the developers. \n\n"
                   "6. ARCHITECT SYNTHESIS: Every technical subheading must contain high-density specification text. Do not leave instructions or placeholders. If a technical detail is missing, provide a standard industry-best-practice recommendation (e.g., for 'Security', suggest OIDC or OAuth2 if not specified). \n"
                   "7. TOTAL CAPTURE: Ensure every field name, business rule, and technical constraint mentioned in the BRD is translated into technical spec. \n"
                   "8. DYNAMIC ARCHITECTURE: Use deep logical numbering (2.1.1, 2.1.2) for clarity. \n\n"
                   "STYLING: Use professional HTML. Center the Title page. \n\n"
                   "9. KEYWORD PARSING: Create specific technical subsections for every module, integration, or rule set mentioned in the BRD or user input. \n\n"
                   "INPUT DATA:\n"
                   "- BRD Context: {brd_content} \n"
                   "- Previous TS Draft: {previous_ts} \n"
                   "- Requirements: {gathered_info} \n"
                   "- Original Project Goal: {project_goal} \n\n"
                   "STRICT INCREMENTAL EDITING: If a previous TS draft is provided, do NOT rewrite it. Only apply the specific changes requested. \n\n"
                   "Output ONLY raw HTML."),
        ("user", "Generate the Final Technical Specification.")
    ])
    
    chain = prompt | doc_llm
    
    try:
        result = chain.invoke({
            "project_goal": state.get('user_goal', "Not provided"),
            "brd_content": state.get('draft_html', ""),
            "gathered_info": state.get('gathered_info', {}),
            "previous_ts": state.get('ts_html', "No previous TS draft exists.")
        })
        
        content = result.content
        final_text = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'text' in item: final_text += item['text']
                elif hasattr(item, 'text'): final_text += item.text
                else: final_text += str(item)
            content = final_text

        html_content = content.replace('\\n', ' ').strip()
        # Robust cleanup for potential JSON leakage
        if html_content.startswith('{') and 'ts' in html_content:
            import re
            match = re.search(r'"(?:ts|content)":\s*"(.*)"\s*}?$', html_content, re.DOTALL)
            if match:
                html_content = match.group(1)

        html_content = html_content.replace('}{ "ts":', '').strip()
        if html_content.endswith('}'):
             if '{' in content and 'ts' in content[:50]:
                html_content = html_content[:-1].strip()
        
        if html_content.startswith("```html"): html_content = html_content[7:-3].strip()
        elif html_content.startswith("```"): html_content = html_content[3:-3].strip()
        
        return {
            **state,
            "ts_html": html_content,
            "conversation_active": False
        }
    except Exception as e:
        print(f"Error in TS generator: {e}")
        return state
