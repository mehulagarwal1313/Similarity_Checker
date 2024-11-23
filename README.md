# Similarity_Checker
This project is a Python-based application designed to analyze and cluster programming code files based on their similarity. It leverages advanced text similarity algorithms, AST (Abstract Syntax Tree) visualization, and graph-based representations to group similar approaches, identify outliers, and highlight innovative coding patterns. 
Key Features:
Code Clustering and Analysis:

Utilizes clustering algorithms like DBSCAN to group code submissions based on structural and functional similarities.
Detects outliers, representing unique or unconventional solutions.
AST Visualization:

Generates graphical representations of the abstract syntax tree (AST) for submitted code files using Graphviz.
Provides insights into the structural components of code for better understanding.
Similarity Matrix Computation:

Computes pairwise similarity scores using techniques like token-based comparison and cosine similarity.
Displays a similarity matrix to compare multiple files at once.
Report Generation:

Creates detailed reports showcasing clusters, outliers, similarity scores, and AST visualizations.
Exports the report in a user-friendly format for further review.
User-Friendly Interface:

Command-line interface to provide inputs and visualize outputs efficiently.
Seamlessly integrates with file systems to process large sets of code files.
Technical Details:
Language: Python
Libraries Used:
Graphviz for AST visualization
Scikit-learn for clustering and similarity computation
Matplotlib/Seaborn for graphical plots (if used for visualization)
External Dependencies:
Graphviz installation for AST rendering.
Purpose: To assist educators and professionals in evaluating code submissions efficiently, while encouraging innovation and diverse approaches to solving problems.
Challenges Addressed:
Clustering large sets of programming code based on functional and structural attributes.
Visualizing code structures in an interpretable format for detailed analysis.
Simplifying the detection of plagiarism or repetitive solutions in programming assignments.
