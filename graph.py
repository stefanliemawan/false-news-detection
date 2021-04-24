from graphviz import Digraph, Graph

dot = Digraph(comment='Test')
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

dot.render('graphs/test.gv', view=True)
