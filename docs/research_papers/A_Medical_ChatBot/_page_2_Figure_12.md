# _page_2_Figure_12.md

**Figure 3 : Separating Hyper Plane By Equation**

**Image Analysis:** The image contains a mathematical formulation and a supporting diagram related to optimizing a Support Vector Machine (SVM).

The main text is:
> To find the optimal hyper plane, we need to solve the following optimization problem:
> $$max \frac{1}{\|w\|}$$
> $$s.t. y_i(w^Tx_i + b) \geq 1, \forall i$$

Below this, a diagram illustrates the SVM margin concept:
*   A solid line represents the **"Decision boundary" (w^Tx = 0)**.
*   A dashed line above it represents the **"Positive" hyperplane (w^Tx = 1)**.
*   A dashed line below it represents the **"Negative" hyperplane (w^Tx = -1)**.
*   The space between these two dashed lines is labeled **"Margin = 2/||w||"**.
*   Data points are shown on both sides of the margin, with those lying directly on the dashed lines likely representing the **Support Vectors**.

The image presents the core SVM optimization objective: to maximize the margin (inversely related to the norm of the weight vector `w`) subject to the constraint that all data points are correctly classified with a functional margin of at least 1.