# GraphSAGE virality analysis

Use `graphsage_predict` when the user asks about spread or virality of imported posts.
Use graph IDs supplied by the application; never invent graph IDs or observed links.
Predict only after graph validation and, for estimated graphs, explicit user acceptance.
Explain viral probability separately from factual support. State the checkpoint's size threshold
and observation window. An estimated graph always means an experimental prediction.
If no checkpoint exists, explain how to import Twitter15/16 and train in the Training view.
For a bundle of unrelated cascades, report each prediction individually; do not sum probabilities.
