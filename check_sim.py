import os
import ast
from difflib import SequenceMatcher
import tokenize
from io import BytesIO
from sklearn.cluster import DBSCAN
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from pathlib import Path
from markdown2 import markdown
from pathlib import Path
# Similarity functions (AST, token, and control flow)
def ast_similarity(code1, code2):
    try:
        tree1, tree2 = ast.parse(code1), ast.parse(code2)
    except SyntaxError:
        return 0.0
    nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
    nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
    return SequenceMatcher(None, nodes1, nodes2).ratio()

def tokenize_code(code):
    tokens = []
    g = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for token in g:
        if token.type == tokenize.NAME:
            tokens.append('IDENTIFIER')
        elif token.type == tokenize.OP:
            tokens.append(token.string)
        elif token.type == tokenize.NUMBER:
            tokens.append('NUMBER')
        elif token.type == tokenize.STRING:
            tokens.append('STRING')
    return tokens

def token_similarity(code1, code2):
    tokens1, tokens2 = tokenize_code(code1), tokenize_code(code2)
    return SequenceMatcher(None, tokens1, tokens2).ratio()

def control_flow_similarity(code1, code2):
    nodes1 = [type(node).__name__ for node in ast.walk(ast.parse(code1))]
    nodes2 = [type(node).__name__ for node in ast.walk(ast.parse(code2))]
    return SequenceMatcher(None, nodes1, nodes2).ratio()

def calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim, weights=(0.4, 0.4, 0.2)):
    return (weights[0] * ast_sim) + (weights[1] * token_sim) + (weights[2] * control_flow_sim)

def load_codes_from_folder(folder_path):
    code_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    codes = {}
    for file in code_files:
        with open(os.path.join(folder_path, file), 'r') as f:
            codes[file] = f.read()
    return codes

def calculate_pairwise_similarities(codes):
    similarity_threshold = 0.7
    files = list(codes.keys())
    similarities = []
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:
                ast_sim = ast_similarity(codes[file1], codes[file2])
                token_sim = token_similarity(codes[file1], codes[file2])
                control_flow_sim = control_flow_similarity(codes[file1], codes[file2])
                overall_sim = calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim)
                if overall_sim > similarity_threshold:
                    similarities.append((file1, file2, overall_sim))
    return similarities

def cluster_codes(similarities, codes, eps=0.5, min_samples=2):
    files = list(codes.keys())
    n = len(files)
    similarity_matrix = np.zeros((n, n))
    for (file1, file2, sim) in similarities:
        i, j = files.index(file1), files.index(file2)
        similarity_matrix[i][j] = similarity_matrix[j][i] = sim
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(1 - similarity_matrix)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(files[idx])
    return clusters, similarity_matrix, files

# Visualization functions
def visualize_similarity_matrix(similarity_matrix, files, output_dir):
    # Ensure diagonal values are 1
    np.fill_diagonal(similarity_matrix, 1)
    
    # Set figure size and create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=files, 
        yticklabels=files, 
        cmap="YlGnBu", 
        cbar=True, 
        square=True  # Ensures cells are square-shaped
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add a title and adjust layout
    plt.title("Code Similarity Heatmap", fontsize=16, pad=20)
    plt.tight_layout()  # Automatically adjust layout to prevent cutoff

    # Save the heatmap
    filename = os.path.join(output_dir, "similarity_matrix.png")
    plt.savefig(filename, dpi=300)  # High resolution for better quality
    plt.close()
    
    return filename

def visualize_ast(code, filename):
    try:
        tree = ast.parse(code)
        graph = graphviz.Digraph(filename=filename)
        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.node(node_id, label=type(node).__name__)
            if parent:
                graph.edge(str(id(parent)), node_id)
            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node)
        add_nodes_edges(tree)
        graph.render(filename=filename, format='png', cleanup=True)
        return str(filename) + '.png'
    except SyntaxError:
        return None

def visualize_tokens(code, filename):
    tokens = tokenize_code(code)
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    
    plt.figure(figsize=(10, 5))
    plt.bar(token_counts.keys(), token_counts.values(), color='skyblue')
    plt.title("Token Frequency")
    plt.xlabel("Token Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filepath = f"{filename}.png"
    plt.savefig(filepath)
    plt.close()
    return filepath

def visualize_control_flow(code, filename):
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()

        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.add_node(node_id, label=type(node).__name__)
            if parent:
                graph.add_edge(str(id(parent)), node_id)
            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node)

        add_nodes_edges(tree)

        pos = nx.spring_layout(graph, k=0.5)  # Adjust k to increase spacing
        labels = nx.get_node_attributes(graph, 'label')
        node_sizes = [1000 if labels[node] == 'FunctionDef' else 300 for node in graph]
        
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos, labels=labels, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10)
        
        filepath = f"{filename}.png"
        plt.savefig(filepath)
        plt.close()
        return filepath
    except SyntaxError:
        return None
def generate_report(clusters, similarities, codes, similarity_matrix, files, output_dir,output_dir2):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dir2 = Path(output_dir2)
    
        

    # Generate the cluster table (Separate HTML)
    cluster_md = """<html>
    <head>
        <title>Cluster Table</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                border: 1px solid black;
                font-family: Arial, sans-serif;
            }
            th, td {
                border: 1px solid black;
                padding: 10px;
                text-align: center;
            }
            thead {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:nth-child(odd) {
                background-color: #ffffff;
            }
            tr.green {
                background-color: #c9f7c9; /* Green for outliers */
            }
            tr.red {
                background-color: #ffcccc; /* Red for the largest cluster */
            }
            tr.yellow {
                background-color: #ffff99; /* Yellow for other clusters */
            }
        </style>
    </head>
    <body>
        <h1>Cluster Overview</h1>
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Files</th>
                    <th>Total Files</th>
                </tr>
            </thead>
            <tbody>
    """

    # Find the cluster with the most members
    max_cluster_size = max(len(files) for files in clusters.values())
    
    for cluster_id, cluster_files in clusters.items():
        # Determine the color based on the number of members in the cluster
        if cluster_id == -1:  # Outliers
            highlight_class = "green"
        elif len(cluster_files) == max_cluster_size:  # Cluster with the most members
            highlight_class = "red"
        else:  # All other clusters
            highlight_class = "yellow"
        
        cluster_md += f"""
                <tr class="{highlight_class}">
                    <td>{'Outliers' if cluster_id == -1 else cluster_id}</td>
                    <td>{', '.join(cluster_files)}</td>
                    <td>{len(cluster_files)}</td>
                </tr>
        """
    cluster_md += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save cluster table to separate HTML file
    cluster_path = output_dir2 / "cluster.html"
    cluster_path.write_text(cluster_md)
    print(f"Cluster table generated at {cluster_path}")

    # Generate the main report
    '''report_md = "# Code Similarity Report\n\n"
    
    # Similarity Matrix Visualization
    sim_matrix_img = visualize_similarity_matrix(similarity_matrix, files, output_dir)
    report_md += f"![Similarity Matrix]({sim_matrix_img})\n\n"

    # Add reference to the cluster table
    report_md += "## Cluster Overview\n\n"
    report_md += f"View the detailed cluster table [here](cluster.html).\n\n"
    '''
    from flask import url_for
    
    from markdown2 import markdown

# Assume similarity_matrix and files are provided
# Assume output_dir and output_dir2 are paths where files are saved

# Initialize the HTML content for the report
    heat_mp = """
    <h1>Code Similarity Report</h1>

    <p style="text-align: center;">
        <img src="{{ url_for('custom_static', filename='similarity_matrix.png') }}" 
             alt="Custom Static Image" 
             style="max-width: 100%; max-height: 100vh; object-fit: contain;" />
    </p>
"""


# Generate the similarity matrix image
    sim_matrix_img = visualize_similarity_matrix(similarity_matrix, files, output_dir)

# Write the HTML file to the output directory
    heat_path = output_dir2 / "heatmap.html"
    heat_path.write_text(heat_mp)

    # Start building the HTML structure for the report
    report_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Similarity Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 1rem;
                background-color: #f9f9f9;
            }
            .container {
                max-width: 900px;
                margin: auto;
                background: #fff;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            h1, h2 {
                text-align: center;
                color: #333;
            }
            .cluster {
                margin-bottom: 1rem;
            }
            .visualization {
                display: none;
                margin-top: 1rem;
            }
            .btn {
                display: inline-block;
                margin: 0.5rem 0;
                padding: 0.5rem 1rem;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 0.5rem 0;
            }
        </style>
        <script>
            function toggleVisualization(id) {
                const section = document.getElementById(id);
                section.style.display = section.style.display === "none" ? "block" : "none";
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Code Similarity Report</h1>
            <h2>Clusters</h2>
    """

    # Iterate over clusters to generate content for each cluster
    for cluster_id, cluster_files in clusters.items():
        cluster_title = "Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
        report_html += f"<h3>{cluster_title}</h3>"
        
        for file in cluster_files:
            sanitized_file = file.replace(" ", "_").replace("/", "_")
            report_html += f"""
            <div class="cluster">
                <p>
                    <strong>{file}</strong>
                    <button class="btn" onclick="toggleVisualization('{sanitized_file}-visualizations')">View Visualizations</button>
                </p>
                <div id="{sanitized_file}-visualizations" class="visualization">
            """
            
            # Add AST Visualization
            ast_img = visualize_ast(codes[file], output_dir / f"{file}_ast")
            if ast_img:
                report_html += f"""
                    <p><strong>AST:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_ast.png') }}}}" alt="AST for {file}">
                """
            
            # Add Token Frequency Visualization
            token_img = visualize_tokens(codes[file], output_dir / f"{file}_tokens")
            if token_img:
                report_html += f"""
                    <p><strong>Token Frequency:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_tokens.png') }}}}" alt="Token Frequency for {file}">
                """
            
            # Add Control Flow Visualization
            cfg_img = visualize_control_flow(codes[file], output_dir / f"{file}_cfg")
            if cfg_img:
                report_html += f"""
                    <p><strong>Control Flow:</strong></p>
                    <img src="{{{{ url_for('custom_static', filename='{sanitized_file}_cfg.png') }}}}" alt="CFG for {file}">
                """
            
            # Close the visualization section
            report_html += "</div></div>"

    # Close the HTML tags to finish the report
    report_html += """
        </div>
    </body>
    </html>
    """

    # Write the HTML report to file
    report_path = output_dir2 / "report.html"
    report_path.write_text(report_html)
    print(f"Report generated at {report_path}")

    
    



'''def main():
    folder_path = os.path.join(os.getcwd(), "codes")
    output_dir = os.path.join(os.getcwd(), "report")
    codes = load_codes_from_folder(folder_path)
    similarities = calculate_pairwise_similarities(codes)
    clusters, similarity_matrix, files = cluster_codes(similarities, codes)
    print("Code Clusters based on Approach:")
    for cluster_id, cluster_files in clusters.items():
        if cluster_id == -1:
            print("Outliers:", cluster_files)
        else:
            print(f"Cluster {cluster_id}: {cluster_files}")
    generate_report(clusters, similarities, codes, similarity_matrix, files, output_dir)

if __name__ == "__main__":
    main()'''
