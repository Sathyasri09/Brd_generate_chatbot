import chainlit as cl
from graph import app_graph
from schema import AgentState
from utils import convert_html_to_docx
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize state in session
@cl.on_chat_start
async def start():
    cl.user_session.set("state", {
        "messages": [],
        "user_goal": "",
        "gathered_info": {},
        "missing_info": [],
        "draft_html": None,
        "ts_html": None,
        "version": 0,
        "conversation_active": True
    })
    
    await cl.Message(
        content="Welcome to the Elite BRD & TS Generator! \n\n"
                "I will interview you to draft a comprehensive **Business Requirement Document (BRD)**. "
                "Once finalized, I can also generate a **Technical Specification (TS)** for your developers. \n"
                "Please tell me about your project to get started."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    state = cl.user_session.get("state")
    
    # 1. Update state with user message
    current_messages = state["messages"] + [{"role": "user", "content": message.content}]
    state["messages"] = current_messages
    
    # 2. Run the graph with session persistence (Asynchronous)
    config = {"configurable": {"thread_id": cl.user_session.get("id")}}
    res_state = await app_graph.ainvoke(state, config=config)
    
    # 3. Process result
    cl.user_session.set("state", res_state)
    
    # --- PHASE 1: Conversation Message Delivery ---
    new_messages = res_state["messages"]
    if len(new_messages) > len(current_messages):
        last_msg = new_messages[-1]
        if last_msg["role"] == "assistant":
            await cl.Message(content=last_msg["content"]).send()
    
    # --- PHASE 2: Document Generation Delivery ---
    # CASE A: A new or updated BRD has been generated
    # logic: if version increased OR if it's the very first doc
    is_initial_brd = res_state.get("draft_html") and not state.get("draft_html")
    is_updated_brd = res_state.get("version", 0) > state.get("version", 0)
    
    if is_initial_brd or is_updated_brd:
        msg_text = "Professional BRD Generation Complete!" if is_initial_brd else "BRD has been updated!"
        await cl.Message(content=f"{msg_text} Creating your document...").send()
        
        draft = res_state.get("draft_html")
        if draft:
            session_id = cl.user_session.get("id")
            filename = f"brd_{session_id[:8]}.docx"
            docx_path = convert_html_to_docx(draft, filename)
            
            elements = [cl.File(name=filename, path=docx_path, display="inline")]
            await cl.Message(content=f"Here is your document: {filename}. \n\nYou can review it and ask for further changes or type **'Generate Technical Specification'**.", elements=elements).send()
        else:
            await cl.Message(content="I apologize, but there was an issue generating the document content. Please try again.").send()

    # CASE B: A Technical Specification has been generated
    elif res_state.get("ts_html") and not state.get("ts_html"):
        await cl.Message(content="Technical Specification (TS) Complete! Creating your document...").send()
        
        ts_content = res_state.get("ts_html")
        if ts_content:
            session_id = cl.user_session.get("id")
            filename = f"ts_{session_id[:8]}.docx"
            ts_path = convert_html_to_docx(ts_content, filename)
            
            elements = [cl.File(name=filename, path=ts_path, display="inline")]
            await cl.Message(content=f"Here is your Technical Specification: {filename}", elements=elements).send()
        else:
            await cl.Message(content="I apologize, but there was an issue generating the technical specification.").send()

    # --- PHASE 3: Fallbacks ---
    else:
        if not res_state.get("user_goal") and len(new_messages) <= len(current_messages):
             await cl.Message(content="Please provide a bit more detail about the project goal.").send()
