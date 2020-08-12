from kaggle_environments import evaluate, make
# from submission import agent
from minimax import agent

env = make("halite", debug=True)

env.run([agent, "random", "random", "random"])
f = open('./index.html', 'w')
f.write(env.render(mode='html', width=400, height=600))
f.close()