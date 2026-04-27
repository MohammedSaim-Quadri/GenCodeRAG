from tree_sitter import Language, Parser
from tree_sitter_javascript import language as javascript_language


def extract_javascript_functions(code: str):
    parser = Parser()
    JS_LANGUAGE = Language(javascript_language())

    parser.language = JS_LANGUAGE

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    functions = []

    def walk(node):
        if node.type == "function_declaration":
            function_name = "anonymous"

            for child in node.children:
                if child.type == "identifier":
                    function_name = code[
                        child.start_byte:child.end_byte
                    ]
                    break

            function_code = code[
                node.start_byte:node.end_byte
            ]

            functions.append({
                "function_name": function_name,
                "code": function_code
            })

        for child in node.children:
            walk(child)

    walk(root)
    return functions