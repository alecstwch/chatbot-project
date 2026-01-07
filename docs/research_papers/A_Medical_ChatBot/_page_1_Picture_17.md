# _page_1_Picture_17.md

**Fig 1 : System Architecture**

**Image Analysis:** The image presents the system architecture of a Medical Chatbot. The architecture includes the following components:

*   **User Android:** Represents the user interface on an Android device, allowing users to "Ask Medical Related Question", "Get Medicine Details", and initiate "Disease Prediction".
*   **Processing:** This section details the chatbot's internal processing steps:
    *   **Voice To Text:** Converts voice input to text.
    *   **NLP:** Natural Language Processing is applied to understand the text.
    *   **Support vector machine(SVM):** SVM is used for disease prediction based on "Training Data" and "Test Data".
    *   **Features Attributes:** Attributes like "Age", "Smoking", and "Heart Rate" are used as input for the SVM.
*   **Google Voice API:** Used for voice-to-text and text-to-voice conversion.
*   **ChatBot API:** Handles communication between the user interface and the database.
*   **MYSQL:** A database for storing and retrieving information.

**Data Flow and Connections:**

The diagram illustrates the flow of information as follows:

1.  The user interacts with the **User Android** app, sending a voice or text query. This interface has a bidirectional connection with the **Processing** unit, allowing it to send the query and receive the final response.
2.  Inside the **Processing** unit, the flow is sequential:
    *   A **Voice To Text** module first converts any speech input to text.
    *   The text is then passed to an **NLP** (Natural Language Processing) module to extract meaning and intent.
    *   For disease prediction tasks, the processed query along with **Features Attributes** (like Age, Smoking status, Heart Rate) are fed into the **Support Vector Machine (SVM)** algorithm. The SVM uses **Training Data** and **Test Data** to make a prediction.
3.  The **Processing** unit utilizes two external APIs:
    *   It interacts with the **Google Voice API** (likely for advanced speech recognition and synthesis).
    *   It communicates with a dedicated **ChatBot API** to handle the logic of fetching information.
4.  Both the **Google Voice API** and the **ChatBot API** have bidirectional access to the **MYSQL** database. They can query the database to retrieve stored information (like medicine details or general medical knowledge) and potentially update it.
5.  The final response from the processing pipeline is sent back through the **ChatBot API** and **Processing** unit to the **User Android** interface for display or spoken output.

This architecture creates an integrated system where user input is processed through NLP and machine learning, supported by external APIs and a central database, to deliver medical information and predictions.