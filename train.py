from kaggle_environments import make

board_size = 5
env = make("halite", debug=True, configuration={"size": board_size, "startingHalite": 1000})

# Training agent in first position (player 1) against the default random agent.
trainer = env.train([None, "random", "random", "random"])

obs = trainer.reset()
for _ in range(100):
    
    action = 0 # Action for the agent being trained.
    obs, reward, done, info = trainer.step(action)
    print(reward)
    if done:
        obs = trainer.reset()