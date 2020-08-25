# Shut-the-Box-Optimization
This code utilizes reinforcement learning to play the game Shut the Box.
The code defines functions that the environment will use to run, and then
uses reinforcement learning to teach itself how to best play the game.

I wrote a description of how this code works and an analysis of my results, 
which is published on medium. It can be found at the following link:
https://medium.com/@cory.langenbach/reinforcement-learning-teaches-program-to-optimize-shut-the-box-game-e1fc964c4d8d

------------------------------------------------------------------------------------------------------------------------------------------

Instructions for setting up the code in PyCharm:
- Clone the repository
- Create a new project in PyCharm
- Add the cloned repository into the new project
- Open a terminal within the PyCharm project
- Navigate to the correct directory: cd Shut-the-Box-Optimization-master/
- Install the required modules: pip install -r requirements.txt

Set up the command line argument:
- To the left of the run button, click on the Shut-the-Box-Optimization dropdown menu, and then on 'Edit Configurations...'
- In the parameters window, type either 'epsilon_greedy' or 'softmax' depending on the strategy you want to use
- The code should be ready to run!
