# Refactor Plan for Complex Agentic Workflow  

## Context  
- Current setup has **two workflows**:  
  - `langgraph/` → simple agentic workflow (working correctly).  
  - `backend/` → complex agentic workflow (not working as expected).  

- Goal: Make the `backend/` complex workflow behave as smoothly as the simple one in `langgraph/`.  

---

## Tasks  

### 1. Input Simplification  
- Remove all extra input fields.  
- Keep **only**:  
  - A **heading**, or  
  - A **single prompt input** from the user.  

### 2. Trace Management  
- Currently, **too many traces** are generated in LangGraph Studio.  
- Keep **minimal, useful traces only**.  

### 3. Workflow Logic  
- **Do not** modify the core workflow logic in `backend/`.  
- **Do not** touch `agent_prompts.yaml`.  

### 4. Tools Refactor  
- Replace direct **function calls** with **LangChain tools** for tool calling.  
- Ensure proper integration into the agentic workflow.  

### 5. Workflow Fix  
- Get the **complex agentic workflow in `backend/` running correctly**,  
- Match the robustness of the `langgraph/` simple workflow.  

---

## Most Important Guidelines  

1. **Coding Standards**  
   - Follow conventions from `qwen.md` file strictly.  

2. **API Endpoints**  
   - Anticipate a frontend in later phases (not now).  
   - Use **LangServe** to create endpoints for end-user interaction.  

3. **Code Cleanup**  
   - Remove **redundant** or **garbage code**.  
   - Keep the repo lean, maintainable, and future-ready.  

---

## Expected Outcomes  

- `backend/` complex workflow runs as reliably as `langgraph/` simple workflow.  
- Cleaner input (just a prompt).  
- Reduced, meaningful traces in LangGraph Studio.  
- Tools properly defined with LangChain, instead of raw function calls.  
- Codebase aligned with `qwen.md` standards.  
- LangServe endpoints in place for future frontend integration.  
- No redundant/unused code.  
