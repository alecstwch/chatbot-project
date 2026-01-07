# _page_4_Figure_14.md

**Fig. 1: System Workflow**

**Image Analysis:** The image is a flowchart illustrating the system workflow for a university FAQ chatbot. The process begins with a "User query". This query is first passed to an "AIML query check" module. The module determines if the query is a "Template based question". If a pattern is found ("Pattern found"), the system retrieves and outputs a "Pattern based answer". If no pattern is found ("Pattern not found"), the system provides a "Default answer". The flowchart depicts a clear decision tree for handling user inquiries within the chatbot system.

**Data Flow and Connections:**
The flowchart shows a simple, linear decision path:
1.  The process starts with the **"User query"**.
2.  This query flows into the **"AIML query check"** component.
3.  From this check, the flow **branches** based on the result:
    *   If the query is identified as a **"Template based question"** (Pattern found), the flow proceeds to provide a **"Pattern based answer"**.
    *   If it is not a template-based question (Pattern not found), the flow proceeds to provide a **"Default answer"**.
4.  Both branches are terminal endpoints, representing the final response delivered to the user.