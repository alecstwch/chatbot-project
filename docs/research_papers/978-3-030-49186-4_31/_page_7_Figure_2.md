# _page_7_Figure_2.md

**Fig. 3.** General chatbot architecture

This diagram illustrates the general architecture of a chatbot. The flow begins with a "User Interface" (labeled as "Mobile user") which sends a "User request" to the "Chatbot". The chatbot contains several components:

*   **Language understanding - User message analysis:** This component extracts "User intent" and "Context information" from the user's request.
*   **Response generation:** This component generates the "Chatbot response" that is sent back to the user interface.
*   **Dialogue Management:** This component interacts with both the "Language understanding" and "Response generation" components.
*   **Action execution / Information retrieval:** This component performs actions or retrieves information based on the user's request.
*   **Data sources:** This component includes a "Knowledge base" and the ability to make a "Web (URL request)".

**Data Flow and Connections:**

The diagram details a sequential and branching data flow:
1.  A **"User request"** flows from the **"User Interface"** into the **"Language understanding - User message analysis"** component.
2.  The analysis component outputs two key pieces of data: **"User intent"** and **"Context information"**. These flow directly into the **"Dialogue Management"** component.
3.  The **"Dialogue Management"** component then sends a request to the **"Action execution / Information retrieval"** component.
4.  The **"Action execution / Information retrieval"** component interacts with two **"Data sources"**: it can query the internal **"Knowledge base"** or make an external **"Web (URL request)"**.
5.  Retrieved information/action results flow back from the **"Action execution / Information retrieval"** component to the **"Dialogue Management"** component.
6.  Finally, the **"Dialogue Management"** component provides the necessary data to the **"Response generation"** component, which formulates the final **"Chatbot response"** sent back to the **"User Interface"**.

The architecture shows a closed loop where the dialogue manager centralizes the flow, using context and intent to decide on actions and ultimately craft a coherent response.