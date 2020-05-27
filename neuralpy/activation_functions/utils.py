def get_activation_details(n_inputs, n_nodes, name, type, layer, keyword_arguments):
    return {
        'n_inputs': n_inputs,
        'n_nodes': n_nodes,
        'name': name,
        'type': type,
        'layer': layer,
        "keyword_arguments": keyword_arguments
    }
